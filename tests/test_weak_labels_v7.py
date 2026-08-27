"""v7 multilingual extractor tests — Turkish/Greek grounded in real report snippets."""

from __future__ import annotations

from rsna_knee.text.weak_labels_v7 import detect_language, extract_label_v7


def _pos(text, label):
    r = extract_label_v7(text, label)
    assert r.value == 1, f"{label}: expected positive, got {r}"


def _neg(text, label):
    r = extract_label_v7(text, label)
    assert r.value == 0 and r.confidence >= 0.6, f"{label}: expected committed negative, got {r}"


def test_language_detection():
    assert detect_language("MRI ΑΡΙΣΤΕΡΟΥ ΓΟΝΑΤΟΣ Χωρίς συλλογή") == "el"
    assert detect_language("SOL DİZ MRG. menisküs normaldir") == "tr"
    assert detect_language("There is a complete ACL tear.") == "other"


def test_turkish_normalcy_negatives():
    _neg("Lateral menisküs ve medial menisküs normaldir.", "Medial Meniscus")
    _neg("Lateral menisküs ve medial menisküs normaldir.", "Lateral Meniscus")
    _neg("Ön çapraz bağ normaldir.", "ACL")
    _neg("Diz eklemi içi sıvı miktarı normal.", "Effusion")


def test_turkish_positives():
    _pos("Belirgin eklem efüzyonu izlendi.", "Effusion")
    _pos("Medial menisküs arka boynunda yırtık izlendi.", "Medial Meniscus")
    _pos("Patellofemoral eklem dejenrasyonu ile uyumlu fokal kıkırdak kayıpları.", "PF OA")


def test_borderline_findings_abstain():
    # Host grades "on the fence" as negative; v7 must NOT assert a positive here.
    for text, label in [
        ("Diz ekleminde minimal mayii artışı görüldü.", "Effusion"),
        ("Μικρό ύδραρθρο σημειούται στην άρθρωση.", "Effusion"),
    ]:
        r = extract_label_v7(text, label)
        assert r.value == 0, f"{label}: borderline should not be positive, got {r}"


def test_greek_negatives_and_positives():
    _neg("Χωρίς ενδαρθρική συλλογή υγρού.", "Effusion")
    _pos("Ύδραρθρο στην άρθρωση του γόνατος.", "Effusion")
    _pos("Κάταγμα της επιγονατίδας.", "Fracture")
    _pos("Ρήξη έσω μηνίσκου.", "Medial Meniscus")


def test_greek_micro_sign_normalization():
    # Real reports use the micro-sign 'µ' (U+00B5) instead of Greek mu 'μ'.
    _pos("Ρήξη έσω µηνίσκου.", "Medial Meniscus")


def test_delegates_other_languages_to_v2():
    _pos("There is a complete ACL tear with discontinuity.", "ACL")
    _pos("Rupture complète du LCA.", "ACL")
