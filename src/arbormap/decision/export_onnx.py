"""
Modelo e entradas para o modo "desenhar área livre" da página (simulação no navegador)

Teste 2.3 (Teresina, 400 × 400 px):
  ONNX fp32 53 MB; pesos em fp16 (com Cast para fp32 na carga) 26,5 MB, diferença máxima de
  0,004 °C em relação ao PyTorch; int8 dinâmico 13,4 MB, mas até 0,27 °C de diferença (descartado).
  onnxruntime-web: WebGPU ≈ 3,5 s por previsão; WASM sem multithread (GitHub Pages não envia
  COOP/COEP) ≈ 30 s. Janelas menores que a cidade não reproduzem o efeito (≈ 20% de erro),
  porque a rede usa o contexto da cena inteira; por isso a previsão é sempre da cidade toda.

Saídas em {out_dir}/:
  urbanheatnet_fp16.onnx   entrada x [1, 4, 400, 400] (normalizada), saída y [1, 1, 400, 400]
  entrada.bin              x da composição seca, float32
  plantavel.bin            fração plantável por classe (praça, calçada, lote livre), uint8 0–255
  valido.bin               pixels com previsão (uint8)
  sim.json                 assinatura de vegetação já normalizada, média/DP da LST, PIXEL_HA
"""

import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from arbormap.decision.areas_candidatas import NDVI_MAX_PLANTIO, Simulador
from arbormap.decision.simulate_urbanheatnet import MODEL_CH, PIXEL_HA, slope_degrees

CLASSES = ("praca", "calcada", "outros")


def pesos_fp16(model_proto, min_size=1024):
    """Guarda os pesos grandes em fp16 e insere Cast para fp32: metade do tamanho, mesma conta."""
    g = model_proto.graph
    novos, casts = [], []
    for init in list(g.initializer):
        a = numpy_helper.to_array(init)
        if a.dtype == np.float32 and a.size > min_size:
            novos.append(numpy_helper.from_array(a.astype(np.float16), init.name + "_h"))
            casts.append(helper.make_node("Cast", [init.name + "_h"], [init.name], to=onnx.TensorProto.FLOAT))
            g.initializer.remove(init)
    g.initializer.extend(novos)
    for c in reversed(casts):
        g.node.insert(0, c)
    return model_proto


def exportar(L, meta, ckpt, fr, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sim = Simulador(L, meta, ckpt)
    tmp = out / "_tmp_fp32.onnx"
    torch.onnx.export(sim.model, sim.x, str(tmp), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    m = pesos_fp16(onnx.load(str(tmp)))
    tmp.unlink()
    (out / "urbanheatnet_fp16.onnx").write_bytes(m.SerializeToString())

    sim.x.numpy().astype("<f4").tofile(out / "entrada.bin")
    dem = np.where(np.isfinite(L["dem"]), L["dem"], np.nanmedian(L["dem"]))
    pode = (sim.valid & (L["artefato_fixo"] < 0.5) & (L["ndvi"] < NDVI_MAX_PLANTIO)
            & (L["ndwi"] < 0.1) & (slope_degrees(dem) <= 15))
    np.stack([np.round(np.clip(fr[c], 0, 1) * pode * 255) for c in CLASSES]).astype(np.uint8).tofile(out / "plantavel.bin")
    sim.valid.astype(np.uint8).tofile(out / "valido.bin")
    mean, std = sim.stats
    alvo = {str(MODEL_CH[c]): float((v - mean[c]) / (std[c] + 1e-6)) for c, v in sim.sig.items()}
    info = {"shape": list(sim.x.shape), "alvo_normalizado": alvo, "lst_media_c": sim.mu, "lst_dp_c": sim.sd,
            "pixel_ha": PIXEL_HA, "classes": list(CLASSES), "ndvi_max_plantio": NDVI_MAX_PLANTIO,
            "modelo": Path(ckpt).name}
    (out / "sim.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return sim, info
