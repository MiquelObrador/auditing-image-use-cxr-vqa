"""Plantilla de prompt para la evaluación zero-shot de MMXU (sin historical records).

Fig. 14 del paper MMXU (Mu et al., Findings of ACL 2025, arXiv:2502.11651),
adaptada a los campos reales de MMXU-test.jsonl.
"""

TEMPLATE = """Here are two chest X-RAY images of the same patient. Previous chest X-RAY is Image-1. And current chest X-RAY is Image-2. Your task is to evaluate the differences between the two images based on the provided question.

Question:
{question}

Options:
A: {option_a}
B: {option_b}
C: {option_c}
D: {option_d}

Answer with the option's letter from the given choices directly."""


def build_prompt(entry: dict) -> str:
    """entry: una fila cruda de MMXU-test.jsonl (con 'question' y 'options').

    Devuelve el texto del prompt, sin imágenes -- cada VLMBaseline decide cómo
    incrusta las dos imágenes junto a este texto en predict().
    """
    options = entry["options"]
    return TEMPLATE.format(
        question=entry["question"],
        option_a=options["A"],
        option_b=options["B"],
        option_c=options["C"],
        option_d=options["D"],
    )
