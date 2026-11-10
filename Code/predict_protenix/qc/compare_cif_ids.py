#!/usr/bin/env python3
"""比较实验 CIF 与 Protenix 预测 CIF 的链 ID 和残基 ID。

默认参数对应第二阶段的 1AFX 冒烟测试。也可以通过命令行参数比较其他结构。
脚本只读取结构文件，不修改 CIF。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import gemmi
except ImportError as exc:
    raise SystemExit(
        "缺少 gemmi。请在已经安装 gemmi 的 protenix_test 容器中运行本脚本。"
    ) from exc


DEFAULT_ORIGINAL = Path(
    "/storage9920/home/tinghao.xia/pdb_data/1afx.cif"
)
DEFAULT_PREDICTED = Path(
    "/storage9920/home/tinghao.xia/Json_data/Smoke_1AFX/Complex_json/"
    "pred_output_1afx_seed_42/1afx/seed_42/predictions/1afx_sample_0.cif"
)

RNA_BASES = {
    "A": "A",
    "C": "C",
    "G": "G",
    "U": "U",
    "ADE": "A",
    "CYT": "C",
    "GUA": "G",
    "URA": "U",
}


@dataclass
class ResidueInfo:
    seq_id: str
    number: int | None
    insertion_code: str
    name: str
    one_letter: str


@dataclass
class ChainInfo:
    chain_id: str
    residue_count: int
    sequence: str
    residues: list[ResidueInfo]


@dataclass
class StructureInfo:
    label: str
    path: str
    model_count: int
    inspected_model_index: int
    chains: list[ChainInfo]


def residue_one_letter(residue_name: str) -> str:
    """将常见 RNA 残基名转换为单字母；未知/修饰残基记为 X。"""
    normalized = residue_name.strip().upper()
    if normalized in RNA_BASES:
        return RNA_BASES[normalized]

    try:
        tabulated = gemmi.find_tabulated_residue(normalized)
        letter = str(tabulated.one_letter_code).strip().upper()
        if len(letter) == 1 and letter.isalpha():
            return letter
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return "X"


def insertion_code(seqid: Any) -> str:
    value = getattr(seqid, "icode", "")
    text = str(value)
    if text in {"\x00", "?", "."}:
        return ""
    return text.strip()


def residue_seq_id(residue: Any) -> str:
    number = getattr(residue.seqid, "num", None)
    code = insertion_code(residue.seqid)
    return f"{number}{code}" if number is not None else str(residue.seqid)


def polymer_residues(chain: Any) -> list[Any]:
    """优先使用 Gemmi 的 polymer 定义，必要时回退到 RNA 残基名。"""
    residues = list(chain.get_polymer())
    if residues:
        return residues
    return [
        residue
        for residue in chain
        if residue_one_letter(residue.name) in {"A", "C", "G", "U"}
    ]


def inspect_structure(
    label: str,
    path: Path,
    model_index: int,
) -> StructureInfo:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} 文件不存在：{path}")

    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"{label} 不包含任何 model：{path}")
    if model_index < 0 or model_index >= len(structure):
        raise IndexError(
            f"{label} 的 model-index={model_index} 越界；"
            f"实际 model 数量为 {len(structure)}"
        )

    chains: list[ChainInfo] = []
    for chain in structure[model_index]:
        residues = polymer_residues(chain)
        if not residues:
            continue

        residue_infos = [
            ResidueInfo(
                seq_id=residue_seq_id(residue),
                number=getattr(residue.seqid, "num", None),
                insertion_code=insertion_code(residue.seqid),
                name=residue.name.strip(),
                one_letter=residue_one_letter(residue.name),
            )
            for residue in residues
        ]
        chains.append(
            ChainInfo(
                chain_id=chain.name,
                residue_count=len(residue_infos),
                sequence="".join(item.one_letter for item in residue_infos),
                residues=residue_infos,
            )
        )

    if not chains:
        raise ValueError(f"{label} 的指定 model 中没有找到 RNA/polymer 链：{path}")

    return StructureInfo(
        label=label,
        path=str(path),
        model_count=len(structure),
        inspected_model_index=model_index,
        chains=chains,
    )


def compare_chains(
    original: StructureInfo,
    predicted: StructureInfo,
) -> tuple[list[dict[str, Any]], list[str]]:
    """先按同名链匹配，再按唯一序列匹配剩余链。"""
    comparisons: list[dict[str, Any]] = []
    warnings: list[str] = []
    unused_predicted = set(range(len(predicted.chains)))

    for original_chain in original.chains:
        candidates = [
            index
            for index in unused_predicted
            if predicted.chains[index].chain_id == original_chain.chain_id
        ]
        match_method = "CHAIN_ID"

        if len(candidates) != 1:
            candidates = [
                index
                for index in unused_predicted
                if predicted.chains[index].sequence == original_chain.sequence
            ]
            match_method = "UNIQUE_SEQUENCE"

        if len(candidates) != 1:
            warnings.append(
                f"原始链 {original_chain.chain_id!r} 无法唯一映射到预测链；"
                f"候选数量={len(candidates)}"
            )
            comparisons.append(
                {
                    "original_chain_id": original_chain.chain_id,
                    "predicted_chain_id": None,
                    "match_method": "UNRESOLVED",
                    "sequence_match": False,
                    "residue_count_match": False,
                    "residue_ids_match": False,
                    "residue_names_match": False,
                    "exact_id_match": False,
                }
            )
            continue

        predicted_index = candidates[0]
        unused_predicted.remove(predicted_index)
        predicted_chain = predicted.chains[predicted_index]

        original_ids = [item.seq_id for item in original_chain.residues]
        predicted_ids = [item.seq_id for item in predicted_chain.residues]
        original_names = [item.name for item in original_chain.residues]
        predicted_names = [item.name for item in predicted_chain.residues]

        sequence_match = original_chain.sequence == predicted_chain.sequence
        count_match = original_chain.residue_count == predicted_chain.residue_count
        ids_match = original_ids == predicted_ids
        names_match = original_names == predicted_names
        chain_id_match = original_chain.chain_id == predicted_chain.chain_id

        comparisons.append(
            {
                "original_chain_id": original_chain.chain_id,
                "predicted_chain_id": predicted_chain.chain_id,
                "match_method": match_method,
                "sequence_match": sequence_match,
                "residue_count_match": count_match,
                "residue_ids_match": ids_match,
                "residue_names_match": names_match,
                "exact_id_match": chain_id_match and ids_match,
            }
        )

    for index in sorted(unused_predicted):
        warnings.append(
            f"预测链 {predicted.chains[index].chain_id!r} 没有对应的原始链"
        )

    return comparisons, warnings


def print_structure(info: StructureInfo) -> None:
    print(f"\n===== {info.label} =====")
    print(f"path: {info.path}")
    print(f"model_count: {info.model_count}")
    print(f"inspected_model_index: {info.inspected_model_index}")
    for chain in info.chains:
        print(f"\nchain_id: {chain.chain_id!r}")
        print(f"residue_count: {chain.residue_count}")
        print(f"sequence: {chain.sequence}")
        print(
            "residues:",
            [(item.seq_id, item.name) for item in chain.residues],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="比较实验 CIF 与 Protenix 预测 CIF 的链 ID、残基 ID 和序列"
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=DEFAULT_ORIGINAL,
        help=f"原始实验 CIF；默认：{DEFAULT_ORIGINAL}",
    )
    parser.add_argument(
        "--predicted",
        type=Path,
        default=DEFAULT_PREDICTED,
        help=f"Protenix 预测 CIF；默认：{DEFAULT_PREDICTED}",
    )
    parser.add_argument(
        "--original-model-index",
        type=int,
        default=0,
        help="原始结构使用的 model 下标，默认 0（第一个 model）",
    )
    parser.add_argument(
        "--predicted-model-index",
        type=int,
        default=0,
        help="预测结构使用的 model 下标，默认 0",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="可选：将比较结果同时保存为 JSON",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        original = inspect_structure(
            "ORIGINAL",
            args.original,
            args.original_model_index,
        )
        predicted = inspect_structure(
            "PREDICTED",
            args.predicted,
            args.predicted_model_index,
        )
        comparisons, warnings = compare_chains(original, predicted)
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_structure(original)
    print_structure(predicted)

    print("\n===== COMPARISON =====")
    for item in comparisons:
        print(
            f"{item['original_chain_id']!r} -> "
            f"{item['predicted_chain_id']!r}; "
            f"method={item['match_method']}; "
            f"sequence_match={item['sequence_match']}; "
            f"residue_count_match={item['residue_count_match']}; "
            f"residue_ids_match={item['residue_ids_match']}; "
            f"residue_names_match={item['residue_names_match']}; "
            f"exact_id_match={item['exact_id_match']}"
        )

    for warning in warnings:
        print(f"WARNING: {warning}")

    all_mapped = bool(comparisons) and all(
        item["predicted_chain_id"] is not None for item in comparisons
    )
    sequences_match = all_mapped and all(
        item["sequence_match"] for item in comparisons
    )
    exact_ids_match = all_mapped and not warnings and all(
        item["exact_id_match"] for item in comparisons
    )

    print("\n===== SUMMARY =====")
    print(f"all_chains_mapped: {all_mapped}")
    print(f"all_sequences_match: {sequences_match}")
    print(f"all_chain_and_residue_ids_match: {exact_ids_match}")

    payload = {
        "original": asdict(original),
        "predicted": asdict(predicted),
        "comparisons": comparisons,
        "warnings": warnings,
        "summary": {
            "all_chains_mapped": all_mapped,
            "all_sequences_match": sequences_match,
            "all_chain_and_residue_ids_match": exact_ids_match,
        },
    }
    if args.json_output:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"json_output: {output}")

    return 0 if all_mapped and sequences_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
