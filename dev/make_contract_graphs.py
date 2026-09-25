"""Generate tiny, real ONNX contract fixtures. NOT embedding-model goldens.

Run: .venv/Scripts/python.exe dev/make_contract_graphs.py
Requires only onnx in the checkout's isolated environment. No network operations.
"""
from pathlib import Path
import onnx
from onnx import TensorProto as T, helper as h

DEST = Path(__file__).resolve().parents[1] / "tests/CommunityToolkit.Embeddings.Onnx.Tests/Fixtures"


def save(name, nodes, inputs, output, initializers=()):
    graph = h.make_graph(nodes, name, inputs, [output], list(initializers))
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=9,
                         producer_name="contract-fixtures-not-model-goldens")
    onnx.checker.check_model(model)
    onnx.save(model, DEST / f"{name}.onnx")


def normal(name, types=False, fixed=False, output_name="last_hidden_state"):
    shape = [2, 3] if fixed else ["batch", "sequence"]
    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if types else [])
    inputs = [h.make_tensor_value_info(n, T.INT64, shape) for n in names]
    nodes = []
    for n in names:
        nodes += [h.make_node("Cast", [n], [n + "_f"], to=T.FLOAT),
                  h.make_node("Unsqueeze", [n + "_f", "axis"], [n + "_u"])]
    if not types:
        nodes += [h.make_node("Mul", ["input_ids_u", "zero"], ["zeros"]),
                  h.make_node("Add", ["zeros", "one"], ["token_type_ids_u"])]
    nodes += [h.make_node("Concat", ["input_ids_u", "attention_mask_u", "token_type_ids_u"],
                         [output_name], axis=2)]
    init = [h.make_tensor("axis", T.INT64, [1], [2])]
    if not types:
        init += [h.make_tensor("zero", T.FLOAT, [], [0]), h.make_tensor("one", T.FLOAT, [], [1])]
    save(name, nodes, inputs, h.make_tensor_value_info(output_name, T.FLOAT, shape + [3]), init)


def invalid(name, fault):
    inputs = [h.make_tensor_value_info("input_ids", T.INT64, ["b", "s"]),
              h.make_tensor_value_info("attention_mask", T.INT64, ["b", "s"])]
    if fault == "missing_ids":
        inputs.pop(0)
    elif fault == "missing_mask":
        inputs.pop(1)
    elif fault == "unexpected":
        inputs.append(h.make_tensor_value_info("position_ids", T.INT64, ["b", "s"]))
    elif fault in ("float_ids", "rank1_ids", "rank3_ids", "float_mask", "rank1_mask", "float_types", "rank1_types"):
        target = "attention_mask" if "mask" in fault else "token_type_ids" if "types" in fault else "input_ids"
        shape = ["b"] if "rank1" in fault else ["b", "s", 1] if "rank3" in fault else ["b", "s"]
        item = h.make_tensor_value_info(target, T.FLOAT if "float" in fault else T.INT64, shape)
        if "types" in fault:
            inputs.append(item)
        else:
            inputs[1 if "mask" in fault else 0] = item
    out_name = "pooled" if fault == "missing_output" else "last_hidden_state"
    shape = [1, 3] if fault == "pooled" else [1, 2, 4] if fault == "dimensions" else [1, 2, 3]
    out_type = T.DOUBLE if fault == "double_output" else T.FLOAT
    size = 1
    for d in shape:
        size *= d
    value = h.make_tensor("constant", out_type, shape, [1] * size)
    save(name, [h.make_node("Constant", [], [out_name], value=value)], inputs,
         h.make_tensor_value_info(out_name, out_type, shape))


def runtime_wrong():
    # Unknown output width until runtime: sum(mask). With two tokens width=2, not 3.
    inputs = [h.make_tensor_value_info(n, T.INT64, ["b", "s"]) for n in ("input_ids", "attention_mask")]
    nodes = [h.make_node("Shape", ["input_ids"], ["shape"]),
             h.make_node("ReduceSum", ["attention_mask"], ["width"], keepdims=0),
             h.make_node("Unsqueeze", ["width", "axis"], ["width1"]),
             h.make_node("Concat", ["shape", "width1"], ["out_shape"], axis=0),
             h.make_node("ConstantOfShape", ["out_shape"], ["last_hidden_state"])]
    save("runtime_dimensions", nodes, inputs,
         h.make_tensor_value_info("last_hidden_state", T.FLOAT, ["b", "s", "hidden"]),
         [h.make_tensor("axis", T.INT64, [1], [0])])

def runtime_wrong_axis(axis):
    inputs = [h.make_tensor_value_info(n, T.INT64, ["b", "s"]) for n in ("input_ids", "attention_mask")]
    nodes = [h.make_node("Shape", ["input_ids"], ["shape"]),
             h.make_node("Add", ["shape", "delta"], ["wrong_shape"]),
             h.make_node("Concat", ["wrong_shape", "width"], ["out_shape"], axis=0),
             h.make_node("ConstantOfShape", ["out_shape"], ["last_hidden_state"])]
    save(f"runtime_axis{axis}", nodes, inputs,
         h.make_tensor_value_info("last_hidden_state", T.FLOAT, ["b", "s", "hidden"]),
         [h.make_tensor("delta", T.INT64, [2], [1, 0] if axis == 0 else [0, 1]),
          h.make_tensor("width", T.INT64, [1], [3])])


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    normal("optional_types")
    normal("required_types", types=True)
    normal("fixed_shape", types=True, fixed=True)
    normal("named_logits", output_name="logits")
    for fault in ("missing_ids", "missing_mask", "unexpected", "float_ids", "rank1_ids", "rank3_ids",
                  "float_mask", "rank1_mask", "float_types", "rank1_types",
                  "missing_output", "pooled", "dimensions", "double_output"):
        invalid(fault, fault)
    runtime_wrong()
    runtime_wrong_axis(0)
    runtime_wrong_axis(1)
    print(f"Generated 21 deterministic ONNX contract fixtures in {DEST}")
