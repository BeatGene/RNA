#Modify_3
UNKNOWN_ATOM_NAME_ID = 0

RNA_ATOM_NAMES = (
      # 磷酸和糖环
      "P", "OP1", "OP2", "OP3",
      "O5'", "C5'", "C4'", "O4'",
      "C3'", "O3'", "C2'", "O2'", "C1'",

      # 碱基环
      "N9", "C8", "N7", "C5", "C6",
      "N1", "C2", "N3", "C4",

      # 碱基环外原子
      "N6", "O6", "N2", "O2", "N4", "O4",
)

ATOM_NAME_TO_ID = {
    atom_name: atom_id
    for atom_id, atom_name in enumerate(
        RNA_ATOM_NAMES,
        start=1,
    )
}

#Modify_4
NUM_ATOM_NAME_TYPES = max(
    ATOM_NAME_TO_ID.values(),
    default=UNKNOWN_ATOM_NAME_ID,
) + 1

UNKNOWN_RESIDUE_TYPE_ID = 4

RNA_RESIDUE_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
}

NUM_RNA_RESIDUE_TYPES = 5
RNAFM_EMBEDDING_DIM = 640

BASE_RING_ATOM_NAMES = (
    "N9", "C8", "N7", "C5", "C6",
    "N1", "C2", "N3", "C4",
)

BASE_ATOM_NAME_IDS = tuple(
    ATOM_NAME_TO_ID[atom_name]
    for atom_name in BASE_RING_ATOM_NAMES
)

ATOM_NAME_ALIASES = {
    "O1P": "OP1",
    "O2P": "OP2",
    "O3P": "OP3",
}


def normalize_atom_name(atom_name: str) -> str:
    atom_name = atom_name.strip().replace("*", "'")
    return ATOM_NAME_ALIASES.get(atom_name, atom_name)
