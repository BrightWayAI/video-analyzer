#!/usr/bin/env python3
"""
Tests for the Stylistic Fingerprint v3 classifier (8 fields).

Run with:  python3 test_fingerprint_v2.py

Tests use real shot descriptions extracted from analyzed videos to ensure
correct classification.  Backward-compatible: also checks v2 alias keys.
"""

import sys
from video_analyzer_mcp.analyze_video import Shot, TranscriptSegment, analyze_stylistic_fingerprint


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_shots(descriptions: list[str], duration_each: float = 2.0) -> list[Shot]:
    """Build Shot objects from a list of visual description strings."""
    shots = []
    for i, desc in enumerate(descriptions):
        shots.append(Shot(
            number=i + 1,
            start_time=i * duration_each,
            end_time=(i + 1) * duration_each,
            visual_description=desc,
        ))
    return shots


def _make_transcript(text: str) -> list[TranscriptSegment]:
    """Wrap a single string into one TranscriptSegment."""
    return [TranscriptSegment(start_time=0, end_time=10, text=text)]


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: "Overdraft Fees: The Charge for Being Broke"
# Expected: Rendering Class = Textured 2D (engraving/watercolor) OR Mixed Media (2D + 3D)
#           (NOT pure 3D — most shots are 2D illustrated/Victorian/watercolor)
# ══════════════════════════════════════════════════════════════════════════
OVERDRAFT_DESCS = [
    "Victorian-era illustration style. Rainy indoor scene. Young man in wet brown coat holds blue bowl, older gentleman in black formal wear.",
    "Victorian illustration style. Indoor scene with heavy rain outside. Young man in wet overcoat holds blue bowl, older gentleman behind counter.",
    "2D Victorian illustration style. Indoor scene with heavy rain outside. Two men in formal 19th century attire.",
    "3D realistic render. Traditional marketplace scene. Wooden ramp filled with metal washers cascading down.",
    "Realistic 3D animation style. Wooden carnival game with metal washers cascading down angled wooden chute.",
    "Vintage illustration style. Victorian-era workshop interior. Three gentlemen in formal coats stand behind workbench.",
    "Vintage illustration style. Workshop interior with wooden workbench. Three men in Victorian-era suits.",
    "2D illustrated animation style. Dark teal textured background. Two large hourglasses with red tops.",
    "2D illustrated animation style. Dark teal textured background. Adult hand dropping coin into left hourglass.",
    "2D illustrated style with cross-hatching. Dark teal background. Two large red-capped hourglasses.",
    "2D illustrated style with crosshatching texture. Dark teal background. Two glass hourglasses with red tops.",
    "2D illustrated art style. Dark teal crosshatched background. Two glass hourglasses with red rims.",
    "2D watercolor illustration style. Victorian-era courtroom interior with wooden paneling.",
    "Vintage illustration style. Old-fashioned bank interior with wooden counters and barred teller windows.",
    "2D illustrated style with cross-hatching detail. Hands dropping coins into cracked ceramic jar.",
    "2D illustrated style with crosshatching. Wooden table surface. Elderly weathered hands cupping coins.",
    "2D illustrated animation style. Indoor scene with blue wall background. Adult hand reaching down.",
    "2D illustrated vintage art style. Hand in dark sleeve sprinkling white powder onto wooden table surface with brass whistle.",
    "2D illustrated style with crosshatch shading. Hand sprinkling white flour or powder onto wooden table.",
    "2D illustrated animation style. Indoor workshop setting with wooden table. Adult hand sprinkling white flour.",
    "Watercolor illustration style. Weathered wooden surface with scattered old coins. Large metal fishing hook.",
    "Watercolor illustration style. Scattered gold coins on weathered surface. Large fishing hook hanging from rope.",
    "2D watercolor illustration style. Weathered ocean floor scene with scattered antique coins. Large rusted fishing hook.",
]

OVERDRAFT_TRANSCRIPT = (
    "Overdraft is the only product that charges you for being broke, and it rarely "
    "feels like a decision. It feels like a surprise. Here's how it happens. You're "
    "a little short, a small charge hits, then another one. And suddenly the fee is "
    "bigger than what you bought. That's the trap. Overdraft isn't a loan you asked "
    "for. It's a penalty that activates when timing goes wrong. Move one. Turn off "
    "overdraft coverage if you can. Move two. Build a tiny buffer. Move three. Set a "
    "low balance alert. And treat it like a smoke alarm."
)


def test_overdraft_rendering_class():
    """Overdraft Fees: should be Textured 2D or Mixed Media, NOT pure 3D."""
    shots = _make_shots(OVERDRAFT_DESCS, duration_each=4.6)
    transcript = _make_transcript(OVERDRAFT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=106.0,
        orientation="vertical", reading_level="Grade 9-10"
    )
    allowed = {
        "Textured 2D (engraving/watercolor)",
        "Mixed Media (2D + 3D)",
    }
    assert result["rendering_class"] in allowed, (
        f"Expected one of {allowed}, got '{result['rendering_class']}'"
    )
    print("  PASS  test_overdraft_rendering_class")


def test_overdraft_narrative_structure():
    """Overdraft Fees: 'Move one/two/three' + 'Here's how' = Problem → Solution."""
    shots = _make_shots(OVERDRAFT_DESCS, duration_each=4.6)
    transcript = _make_transcript(OVERDRAFT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=106.0,
        orientation="vertical", reading_level="Grade 9-10"
    )
    allowed = {"Problem → Solution", "Listicle / Tips"}
    assert result["narrative_structure"] in allowed, (
        f"Expected one of {allowed}, got '{result['narrative_structure']}'"
    )
    print("  PASS  test_overdraft_narrative_structure")


def test_overdraft_tonal():
    """Overdraft Fees: Victorian/moody + moral framing = Dark Editorial / Satirical."""
    shots = _make_shots(OVERDRAFT_DESCS, duration_each=4.6)
    transcript = _make_transcript(OVERDRAFT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=106.0,
        orientation="vertical", reading_level="Grade 9-10"
    )
    assert result["tonal_positioning"] == "Dark Editorial / Satirical", (
        f"Expected 'Dark Editorial / Satirical', got '{result['tonal_positioning']}'"
    )
    print("  PASS  test_overdraft_tonal")


def test_overdraft_world_type():
    """Overdraft Fees: World Type must NOT be Real-World Literal."""
    shots = _make_shots(OVERDRAFT_DESCS, duration_each=4.6)
    transcript = _make_transcript(OVERDRAFT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=106.0,
        orientation="vertical", reading_level="Grade 9-10"
    )
    assert result["world_type"] != "Real-World Literal", (
        f"Expected NOT 'Real-World Literal', got '{result['world_type']}'"
    )
    print("  PASS  test_overdraft_world_type")


def test_overdraft_abstraction():
    """Overdraft Fees: Textured 2D + real environments = Stylized Realism (2)."""
    shots = _make_shots(OVERDRAFT_DESCS, duration_each=4.6)
    transcript = _make_transcript(OVERDRAFT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=106.0,
        orientation="vertical", reading_level="Grade 9-10"
    )
    # Textured 2D with real-world environments should be 2 or 3
    assert result["visual_abstraction_index"].startswith(("2", "3")), (
        f"Expected abstraction 2 or 3, got '{result['visual_abstraction_index']}'"
    )
    print("  PASS  test_overdraft_abstraction")


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: "RENTING VS BUYING what's better"
# Expected: Rendering Class = Stylized 3D
#           Camera/Editing = Social Vertical Punch
# ══════════════════════════════════════════════════════════════════════════
RENT_BUY_DESCS = [
    "3D animated style. Modern apartment interior with floor-to-ceiling windows showing city skyline at sunset.",
    "3D animated style. Dark blue atmospheric background. Young boy with brown messy hair wearing casual clothes.",
    "3D animated style. Suburban house with visible cracks in yellow walls, dark roof.",
    "3D rendered style. Dark indoor setting with torn calendar pages floating above.",
    "3D animated style. Suburban neighborhood at sunset with houses in background. Young boy.",
    "3D animated style. Indoor home setting with warm beige walls. Teenage boy with brown hair.",
    "3D animated style. Indoor home setting with warm beige walls. Teenage boy with brown hair smiling.",
    "3D rendered style. Clean white background. Left side: pile of green dollar bills. Right side: golden pocket watch.",
]

RENT_BUY_TRANSCRIPT = (
    "The eternal question, buy a home or rent, most people say renting is the "
    "dumbest thing ever. But let's look at it without stereotypes. You're just "
    "making your landlord rich. Buying a home can also become a black hole for "
    "money. Taxes, repairs, and you're in the red. Here's the secret. It's about "
    "how long you stay. Less than five years, renting is better. More than five "
    "years. Buying makes sense. So renting isn't always evil. And buying isn't "
    "always smart. Subscribe and follow for more."
)


def test_rent_buy_rendering_class():
    """RENTING VS BUYING: all shots are 3D animated => Stylized 3D."""
    shots = _make_shots(RENT_BUY_DESCS, duration_each=5.25)
    transcript = _make_transcript(RENT_BUY_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=42.0,
        orientation="vertical", reading_level="Grade 8-10"
    )
    assert result["rendering_class"] == "Stylized 3D", (
        f"Expected 'Stylized 3D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_rent_buy_rendering_class")


def test_rent_buy_camera_editing():
    """RENTING VS BUYING: vertical 9:16 + CTA => Social Vertical Punch."""
    shots = _make_shots(RENT_BUY_DESCS, duration_each=5.25)
    transcript = _make_transcript(RENT_BUY_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=42.0,
        orientation="vertical", reading_level="Grade 8-10"
    )
    assert result["camera_editing"].startswith("Social Vertical Punch"), (
        f"Expected 'Social Vertical Punch...', got '{result['camera_editing']}'"
    )
    print("  PASS  test_rent_buy_camera_editing")


def test_rent_buy_world_type():
    """RENTING VS BUYING: apartment + suburban house = Stylized Real-World."""
    shots = _make_shots(RENT_BUY_DESCS, duration_each=5.25)
    transcript = _make_transcript(RENT_BUY_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=42.0,
        orientation="vertical", reading_level="Grade 8-10"
    )
    assert result["world_type"] == "Stylized Real-World", (
        f"Expected 'Stylized Real-World', got '{result['world_type']}'"
    )
    print("  PASS  test_rent_buy_world_type")


def test_rent_buy_narrative():
    """RENTING VS BUYING: 'most people say'+'without stereotypes' = Myth-Busting."""
    shots = _make_shots(RENT_BUY_DESCS, duration_each=5.25)
    transcript = _make_transcript(RENT_BUY_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=42.0,
        orientation="vertical", reading_level="Grade 8-10"
    )
    allowed = {"Myth-Busting / Contrast", "Problem → Solution"}
    assert result["narrative_structure"] in allowed, (
        f"Expected one of {allowed}, got '{result['narrative_structure']}'"
    )
    print("  PASS  test_rent_buy_narrative")


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: "What Is Debt Explained in 30 Seconds"
# Expected: Rendering Class = Minimalist Line Art 2D
# ══════════════════════════════════════════════════════════════════════════
DEBT_DESCS = [
    "2D black and white cartoon style. Simple stick figure businessman in suit and tie struggling to carry enormous wooden crate labeled DEBT.",
    "2D black and white cartoon style. Simple stick figure businessman with round head, angry expression, smoking cigar, wearing black tie.",
    "2D black line art cartoon style. Simple white textured background. Distressed stick figure businessman in black tie, crying.",
    "2D minimalist cartoon style. Textured white paper background. Simple stick figure person with round head, neutral expression, holding money.",
    "2D black line art cartoon style. Simple stick figure character with round smiling head holding a brick, standing next to partially built brick wall.",
    "2D cartoon stick figure animation style. Crumpled white paper textured background. Simple black stick figure.",
    "2D minimalist line art style. Simple stick figure businessman with tie and small hair tuft, smiling, standing beside oversized credit card.",
    "Simple black line drawing on crumpled paper texture. Stick figure character with round smiling head holding brick, standing near house.",
]

DEBT_TRANSCRIPT = (
    "What is debt? It's money you owe. There's good debt, like a mortgage to buy a "
    "house. There's bad debt, like credit card debt from impulse buys. And there's "
    "the costly truth. Interest. The longer you owe, the more you pay."
)


def test_debt_rendering_class():
    """What Is Debt: stick figures + black & white => Minimalist Line Art 2D."""
    shots = _make_shots(DEBT_DESCS, duration_each=3.75)
    transcript = _make_transcript(DEBT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 4-5"
    )
    assert result["rendering_class"] == "Minimalist Line Art 2D", (
        f"Expected 'Minimalist Line Art 2D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_debt_rendering_class")


def test_debt_world_type():
    """What Is Debt: paper texture backgrounds => Abstract Concept Space."""
    shots = _make_shots(DEBT_DESCS, duration_each=3.75)
    transcript = _make_transcript(DEBT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 4-5"
    )
    assert result["world_type"] == "Abstract Concept Space", (
        f"Expected 'Abstract Concept Space', got '{result['world_type']}'"
    )
    print("  PASS  test_debt_world_type")


def test_debt_character_strategy():
    """What Is Debt: stick figures without continuity = None or Single Narrator, NOT Ensemble."""
    shots = _make_shots(DEBT_DESCS, duration_each=3.75)
    transcript = _make_transcript(DEBT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 4-5"
    )
    assert result["character_strategy"] != "Ensemble Cast", (
        f"Expected NOT 'Ensemble Cast', got '{result['character_strategy']}'"
    )
    print("  PASS  test_debt_character_strategy")


def test_debt_tonal():
    """What Is Debt (30s): vertical + short-form + CTA => Gen Z Social / Snappy."""
    shots = _make_shots(DEBT_DESCS, duration_each=3.75)
    transcript = _make_transcript(DEBT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 4-5"
    )
    assert result["tonal_positioning"] == "Gen Z Social / Snappy", (
        f"Expected 'Gen Z Social / Snappy', got '{result['tonal_positioning']}'"
    )
    print("  PASS  test_debt_tonal")


def test_debt_abstraction():
    """What Is Debt: stick figures on paper = High Abstraction (4)."""
    shots = _make_shots(DEBT_DESCS, duration_each=3.75)
    transcript = _make_transcript(DEBT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 4-5"
    )
    assert result["visual_abstraction_index"].startswith(("4", "5")), (
        f"Expected abstraction 4 or 5, got '{result['visual_abstraction_index']}'"
    )
    print("  PASS  test_debt_abstraction")


def test_debt_narrative():
    """What Is Debt: 'What is debt? It's money you owe' = Direct Explanation."""
    shots = _make_shots(DEBT_DESCS, duration_each=3.75)
    transcript = _make_transcript(DEBT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 4-5"
    )
    assert result["narrative_structure"] == "Direct Explanation", (
        f"Expected 'Direct Explanation', got '{result['narrative_structure']}'"
    )
    print("  PASS  test_debt_narrative")


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: "Upstairs, downstairs: The life of a British maid" (TED-Ed)
# Expected: Tone = Institutional / Formal, NOT Dark Editorial
# ══════════════════════════════════════════════════════════════════════════
BRITISH_MAID_DESCS = [
    "2D animated illustration style. Title card with TED-Ed logo. Stylized Victorian-era manor house exterior.",
    "2D illustrated animation. Interior of grand Victorian manor house. Elegant staircase with ornate railing. Maid in black dress with white apron carrying tray.",
    "2D animated style. Kitchen scene in Victorian household. Maid scrubbing floor on hands and knees. Copper pots hanging from ceiling.",
    "2D illustrated. Cross-section view of Victorian manor showing upstairs parlor and downstairs servants quarters. Historical architectural detail.",
    "2D animated illustration. Morning routine. Maid lighting fire in bedroom fireplace before dawn. Candle illumination.",
    "2D illustrated style. Dining room scene. Formal table setting. Butler and maid serving aristocratic family in period attire.",
    "2D animated. Laundry room below stairs. Maid hand-washing linens in large copper basin. Steam rising.",
    "2D illustrated. Servants hall. Multiple servants eating meal at long wooden table. Head housekeeper at end.",
    "2D animated illustration style. Evening scene. Maid turning down bedsheets in master bedroom. Oil lamp on nightstand.",
    "2D illustrated. Wide shot of manor exterior at dusk. Stylized clouds. TED-Ed end card with logo.",
]

BRITISH_MAID_TRANSCRIPT = (
    "In the 19th century, domestic service was one of the largest employers in Britain. "
    "A young woman entering service as a maid of all work faced grueling days that began "
    "before dawn. During this era, the household hierarchy was rigid. Historians estimate "
    "that by 1900, over one million women worked as domestic servants. The maid's duties "
    "ranged from lighting fires to scrubbing floors, serving meals, and attending to every "
    "need of the family upstairs. Life below stairs was a world apart from the elegance above."
)


def test_british_maid_tonal():
    """British Maid: TED-Ed historical documentary = Institutional / Formal."""
    shots = _make_shots(BRITISH_MAID_DESCS, duration_each=18.0)
    transcript = _make_transcript(BRITISH_MAID_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=180.0,
        orientation="horizontal", reading_level="Grade 10-12"
    )
    assert result["tonal_positioning"] == "Institutional / Formal", (
        f"Expected 'Institutional / Formal', got '{result['tonal_positioning']}'"
    )
    print("  PASS  test_british_maid_tonal")


def test_british_maid_not_dark_editorial():
    """British Maid: must NOT be Dark Editorial despite Victorian visuals."""
    shots = _make_shots(BRITISH_MAID_DESCS, duration_each=18.0)
    transcript = _make_transcript(BRITISH_MAID_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=180.0,
        orientation="horizontal", reading_level="Grade 10-12"
    )
    assert result["tonal_positioning"] != "Dark Editorial / Satirical", (
        f"Expected NOT 'Dark Editorial / Satirical', got '{result['tonal_positioning']}'"
    )
    print("  PASS  test_british_maid_not_dark_editorial")


def test_british_maid_narrative():
    """British Maid: '19th century'+'historians'+'by 1900' = Historical / Chronological."""
    shots = _make_shots(BRITISH_MAID_DESCS, duration_each=18.0)
    transcript = _make_transcript(BRITISH_MAID_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=180.0,
        orientation="horizontal", reading_level="Grade 10-12"
    )
    assert result["narrative_structure"] == "Historical / Chronological", (
        f"Expected 'Historical / Chronological', got '{result['narrative_structure']}'"
    )
    print("  PASS  test_british_maid_narrative")


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: Synthetic — 3D Pixar-style with mascot
# Validates mascot detection + Stylized 3D + Fictional Metaphor Universe
# ══════════════════════════════════════════════════════════════════════════
MASCOT_DESCS = [
    "3D animated style. Jungle clearing. Friendly monkey mascot in red vest waves at camera. Bright sunlight.",
    "3D animated style. Banana tree grove. Monkey mascot holding bunch of bananas, pointing at them excitedly.",
    "3D rendered. Monkey mascot at wooden market stall exchanging bananas for golden coins.",
    "3D animated style. Monkey mascot sitting on rock, explaining with hand gestures. Waterfall in background.",
    "3D animated style. Monkey mascot high-fiving a young duck character. Colorful jungle flowers.",
]

MASCOT_TRANSCRIPT = (
    "Hi everyone! Max the monkey explains how money works! When you save bananas, "
    "they grow into more bananas. Subscribe and follow for more."
)


def test_mascot_rendering():
    """3D mascot video: Rendering Class = Stylized 3D."""
    shots = _make_shots(MASCOT_DESCS, duration_each=6.0)
    transcript = _make_transcript(MASCOT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 2-3"
    )
    assert result["rendering_class"] == "Stylized 3D", (
        f"Expected 'Stylized 3D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_mascot_rendering")


def test_mascot_character_strategy():
    """3D mascot video: monkey appears in every shot => Mascot-Led."""
    shots = _make_shots(MASCOT_DESCS, duration_each=6.0)
    transcript = _make_transcript(MASCOT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 2-3"
    )
    assert result["character_strategy"].startswith("Mascot-Led"), (
        f"Expected 'Mascot-Led...', got '{result['character_strategy']}'"
    )
    print("  PASS  test_mascot_character_strategy")


def test_mascot_world_type():
    """3D mascot video: jungle + bananas-as-money => Fictional Metaphor Universe."""
    shots = _make_shots(MASCOT_DESCS, duration_each=6.0)
    transcript = _make_transcript(MASCOT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 2-3"
    )
    assert result["world_type"] == "Fictional Metaphor Universe", (
        f"Expected 'Fictional Metaphor Universe', got '{result['world_type']}'"
    )
    print("  PASS  test_mascot_world_type")


def test_mascot_tonal():
    """3D mascot video: grade 2-3 + anthropomorphic animals => Child-Friendly."""
    shots = _make_shots(MASCOT_DESCS, duration_each=6.0)
    transcript = _make_transcript(MASCOT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 2-3"
    )
    assert result["tonal_positioning"] == "Child-Friendly / Whimsical", (
        f"Expected 'Child-Friendly / Whimsical', got '{result['tonal_positioning']}'"
    )
    print("  PASS  test_mascot_tonal")


def test_mascot_abstraction():
    """3D mascot: Stylized 3D with environments = Stylized Realism (2)."""
    shots = _make_shots(MASCOT_DESCS, duration_each=6.0)
    transcript = _make_transcript(MASCOT_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=30.0,
        orientation="vertical", reading_level="Grade 2-3"
    )
    assert result["visual_abstraction_index"].startswith("2"), (
        f"Expected abstraction 2, got '{result['visual_abstraction_index']}'"
    )
    print("  PASS  test_mascot_abstraction")


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: Synthetic — Data/charts-heavy explainer
# ══════════════════════════════════════════════════════════════════════════
DATA_DESCS = [
    "Clean white background. Large bar chart showing interest rates over time. Blue and green bars.",
    "White background. Pie chart showing budget breakdown. Percentage labels overlaid. Bold title card.",
    "Solid white background. Formula: P = C × (1+r)^n displayed in large text. Calculator icon.",
    "White background. Infographic showing mortgage vs rent comparison chart. Dollar icons.",
    "Clean slide with logo at bottom. Data visualization showing compound interest curve.",
]

DATA_TRANSCRIPT = (
    "Let's understand compound interest using this formula. For example, "
    "if you invest one thousand dollars at five percent, the chart shows "
    "exponential growth over time."
)


def test_data_rendering():
    """Data-heavy video: no 3D/2D character cues => Flat 2D by default."""
    shots = _make_shots(DATA_DESCS, duration_each=10.0)
    transcript = _make_transcript(DATA_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=50.0,
        orientation="horizontal", reading_level="Grade 8-10"
    )
    # Charts on white backgrounds
    assert result["rendering_class"] == "Flat 2D", (
        f"Expected 'Flat 2D', got '{result['rendering_class']}'"
    )
    print("  PASS  test_data_rendering")


def test_data_world_type():
    """Data-heavy video: white backgrounds + charts => Data/Presentation Space."""
    shots = _make_shots(DATA_DESCS, duration_each=10.0)
    transcript = _make_transcript(DATA_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=50.0,
        orientation="horizontal", reading_level="Grade 8-10"
    )
    assert result["world_type"] == "Data/Presentation Space", (
        f"Expected 'Data/Presentation Space', got '{result['world_type']}'"
    )
    print("  PASS  test_data_world_type")


def test_data_narrative():
    """Data-heavy video: charts/formulas dominate => Data-Driven / Analytical."""
    shots = _make_shots(DATA_DESCS, duration_each=10.0)
    transcript = _make_transcript(DATA_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=50.0,
        orientation="horizontal", reading_level="Grade 8-10"
    )
    assert result["narrative_structure"] == "Data-Driven / Analytical", (
        f"Expected 'Data-Driven / Analytical', got '{result['narrative_structure']}'"
    )
    print("  PASS  test_data_narrative")


def test_data_abstraction():
    """Data-heavy video: pure charts/text on white = Maximum Abstraction (5)."""
    shots = _make_shots(DATA_DESCS, duration_each=10.0)
    transcript = _make_transcript(DATA_TRANSCRIPT)
    result = analyze_stylistic_fingerprint(
        shots, transcript, duration=50.0,
        orientation="horizontal", reading_level="Grade 8-10"
    )
    assert result["visual_abstraction_index"].startswith("5"), (
        f"Expected abstraction 5, got '{result['visual_abstraction_index']}'"
    )
    print("  PASS  test_data_abstraction")


# ══════════════════════════════════════════════════════════════════════════
# Ensure all 8 fields are always populated (no empty/None)
# ══════════════════════════════════════════════════════════════════════════
def test_all_fields_populated():
    """Every field must return a non-empty string, never None."""
    # Use minimal input
    shots = _make_shots(["Simple scene with a boy." for _ in range(3)])
    transcript = _make_transcript("Hello world.")
    result = analyze_stylistic_fingerprint(shots, transcript, duration=10.0)
    required_keys = [
        "rendering_class", "world_type", "character_strategy",
        "narrative_structure", "visual_abstraction_index",
        "visual_density", "camera_editing", "tonal_positioning",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        assert result[key], f"Empty value for key: {key}"
        assert isinstance(result[key], str), f"Non-string value for key: {key}"
    print("  PASS  test_all_fields_populated")


def test_v2_backward_compat():
    """v2 backward-compat: metaphor_mode key must still exist."""
    shots = _make_shots(["Simple scene with a boy." for _ in range(3)])
    transcript = _make_transcript("Hello world.")
    result = analyze_stylistic_fingerprint(shots, transcript, duration=10.0)
    assert "metaphor_mode" in result, "Missing v2 compat key: metaphor_mode"
    assert result["metaphor_mode"], "Empty v2 compat key: metaphor_mode"
    print("  PASS  test_v2_backward_compat")


# ── Run all tests ────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        # Overdraft Fees tests
        test_overdraft_rendering_class,
        test_overdraft_narrative_structure,
        test_overdraft_tonal,
        test_overdraft_world_type,
        test_overdraft_abstraction,
        # British Maid tests
        test_british_maid_tonal,
        test_british_maid_not_dark_editorial,
        test_british_maid_narrative,
        # Renting vs Buying tests
        test_rent_buy_rendering_class,
        test_rent_buy_camera_editing,
        test_rent_buy_world_type,
        test_rent_buy_narrative,
        # What Is Debt tests
        test_debt_rendering_class,
        test_debt_world_type,
        test_debt_character_strategy,
        test_debt_tonal,
        test_debt_abstraction,
        test_debt_narrative,
        # Mascot / 3D Fictional tests
        test_mascot_rendering,
        test_mascot_character_strategy,
        test_mascot_world_type,
        test_mascot_tonal,
        test_mascot_abstraction,
        # Data-heavy tests
        test_data_rendering,
        test_data_world_type,
        test_data_narrative,
        test_data_abstraction,
        # General
        test_all_fields_populated,
        test_v2_backward_compat,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed / {len(tests)} total")
    if failed:
        print(f"{failed} FAILED")
    sys.exit(1 if failed else 0)


# ══════════════════════════════════════════════════════════════════════════
# FIXTURE: Big Buck Bunny trailer (Blender Foundation, CC-BY), described by
# the hosted pipeline. Nine of 27 shots are text title cards; that must not
# make a 3D cartoon a "Data/Presentation Space".
# ══════════════════════════════════════════════════════════════════════════
TRAILER_DESCS = [
    "Minimalist flat title card, plain white background. Centered bold uppercase humanist sans-serif text in two lines: \"THE PEACH OPEN MOVIE PROJECT\" in bright cyan-blue, \"PRESENTS\" below in slightly darker teal-blue with a subtle gradient. No characters or other elements. Clean, even lighting. Wide 16:9 frame.",
    "3D animated render, Pixar-style. Sunlit forest meadow with a lush grassy mound, a large tree with exposed gnarled roots on top, and a dark burrow hole beside small gray rocks. Purple wildflowers dot the bright green grass. Soft golden backlight, dappled shadows, dense leafy background. Wide establishing shot.",
    "Minimalist flat title card, 16:9. Pure white background with no other elements. Centered text \"ONE BIG RABBIT\" in bold, wide, uppercase sans-serif letters (Helvetica Black style). Cyan-blue vertical gradient fill, brighter at top and darker teal-blue at bottom. Clean, crisp edges, even flat lighting.",
    "3D animated render, Pixar-style CGI. Sunlit green meadow with long yellow-green grass, distant blue-purple hills, soft pink clouds in a lavender sky, leafy branch in top-left corner. Giant chubby gray-white rabbit stands upright, one ear up and one flopped, mouth slightly open, gazing up curiously. Warm golden lighting. Medium-wide shot.",
    "Flat minimalist title card, 16:9. Plain pure white background. Centered bold uppercase sans-serif text reading \"THREE RODENTS\" in a cyan-to-blue vertical gradient, wide humanist letterforms with a soft glossy finish. No characters, no other elements, even bright lighting, clean and crisp.",
    "3D animated CGI render, Pixar-style. Sunny meadow at dusk with pink sky, distant blue hills, lone green tree. A brown flying squirrel clings to a big tree trunk, smirking with wide blue eyes. A reddish rodent mid-leap with buck teeth, holding a spiky green fruit. A chubby gray chinchilla sits in the grass looking up. Soft dappled light.",
    "3D animated render, Pixar-style. Sunlit green meadow with tall yellow-green grass, distant hazy blue hills, pink-peach clouds in a lavender sky, leafy branch in upper left corner. Giant chubby white rabbit stands upright, hand on hip, long pink ears, heavy-lidded annoyed expression, pursed lips. Soft warm backlighting. Medium wide shot.",
    "3D animated render, Pixar-style. Sunlit green meadow with a large tree trunk and distant blue hills. A tan flying squirrel clings to the trunk, wide-eyed and grinning. A red-brown squirrel beside it laughs, eyes shut, buck tooth showing. A chubby gray chinchilla sits in the grass, eyes closed, smiling. Dappled warm light, pink sky.",
    "3D animated render, Pixar-style CGI. Giant plump white rabbit with fluffy fur, long floppy ears, and a round belly stands in a green meadow, mouth slightly open, eyes wide and glancing sideways in a startled, bewildered expression. Pastel lavender sky, pink clouds, distant blue hills. Soft dusk lighting. Medium shot.",
    "Flat 2D motion graphics title card. Plain white background, no characters or objects. Centered single line of bold, uppercase, wide-set serif-sans text reading \"AND ONE GIANT PAYBACK\" in cyan-blue with a subtle lighter gradient. Soft glow edges, even flat lighting, clean minimalist composition, generous empty space around the text.",
    "Pure black frame, completely empty. Solid uniform black screen with no characters, objects, textures, or gradients. No light sources, total darkness, flat digital black, 16:9 horizontal composition, like a fade-to-black transition frame in an animated video.",
    "3D animated render, Pixar-style CGI. Extreme close-up of a giant, chubby white rabbit with soft fur, narrowed eyes and a determined smirk, aiming a homemade bamboo bow. One paw pulls a green vine string toward his face. Blurred lush jungle foliage behind, soft diffused daylight.",
    "3D animated CGI render. Sunlit forest clearing with dirt path, green ferns, purple flowers, red mushroom. Three anthropomorphic rodents: lanky reddish-brown rabbit-like creature clutching a nut, plump gray chinchilla sitting, and tan squirrel with bushy tail mid-motion, mouth open. Dappled light, soft depth of field, whimsical mood.",
    "3D animated CGI render, stylized cartoon. Sunlit forest clearing with dappled light, green grass, ferns, purple flowers, red mushroom. Three rodents: lanky rust-red rabbit-like creature with droopy eyelids holding a nut, plump gray chinchilla behind, tan-and-white squirrel with bushy tail in foreground, wide-eyed, mid-stride. Medium shot.",
    "3D animated CGI render, dim moody forest clearing at dusk with dirt ground, tall grass, ferns, purple flowers, red mushroom. Three woodland critters: lanky reddish-brown rabbit-squirrel with buck teeth clutching a nut, chubby gray rodent holding an acorn, and a tan-and-white chipmunk in foreground, back turned, eyes shut. Dark, shadowy, low-key lighting. Medium shot.",
    "3D animated render. Dim forest floor at dusk with blurred green grass blades framing both sides. Extreme close-up of a reddish-brown cartoon rabbit's face, wide-angle distortion, mismatched eyes (one wide green, one droopy half-lidded red), pink nose, large buck teeth, long whiskers. Dark, moody, low-key lighting.",
    "3D animated CGI render. Extreme close-up of a goofy reddish-brown cartoon rabbit with fluffy fur, mismatched wide eyes (one green iris, one red-brown, droopy lids), pink nose, single large buck tooth, ears flared sideways, sly lopsided grin. Blurred dirt forest path, green grass blades, soft natural daylight.",
    "3D animated render, dark cartoon style. Extreme close-up of a reddish-brown rabbit's face peeking out of a burrow at night, one wide eye and one squinting eye glancing sideways, large white buck teeth, long whiskered cheek fur, tall ears. Dim, moody low-key lighting, blurred green grass blades at the frame edges.",
    "3D animated render, stylized CGI. Lush sunlit forest floor with vibrant green grass, dirt mound, mossy gray rocks, hanging twisted vines, and yellow-green leaves in the foreground. A small gray chinchilla-like rodent with big pink ears sits upright, clutching a brown chestnut and smiling contentedly. Soft golden backlight, wide shot.",
    "Flat minimalist motion graphic title card. Solid medium-gray background, no characters or objects. Centered bold, uppercase sans-serif text reading \"GET READY\" in teal-blue with a subtle darker gradient toward the bottom of the letters. Soft, even lighting, clean and simple mood. Wide, centered composition.",
    "Flat minimalist motion graphic title card. Plain white background with no other elements. Centered bold uppercase sans-serif text reading \"GET READY\" in a cyan-to-teal blue vertical gradient with a subtle light edge highlight. Clean, bright, even lighting. Wide 16:9 composition.",
    "3D animated CGI render, stylized cartoon. Mischievous brown flying squirrel with wide bulging eyes, bared buck tooth, and menacing grin raises a gray rock overhead in one paw. A glowing pink-purple butterfly perches on a pale boulder at right. Soft blue-to-peach pastel sky, ocean horizon, warm diffused light. Medium shot.",
    "3D animated render, Pixar-style. Sunny green meadow with a leafy bright-green tree and distant lavender hills under a pastel blue sky with fluffy pink-tinged clouds. Giant chubby gray-white rabbit with long upright ears stands facing the camera, wide startled eyes, small open mouth. Soft warm lighting. Medium shot.",
    "Flat 2D title card logo on a plain white background. Bold, chunky sans-serif text \"Big Buck\" above \"BUNNY\" in cyan-to-teal blue gradient. A small blue peach with a leaf sits atop the \"i\" in Big. A white butterfly silhouette overlaps the \"k\" and the \"N\". Clean, centered, no characters.",
    "3D animated CGI film style. Bright blue sky with soft peach clouds. On the left, a heavily motion-blurred brown and gray donkey-like creature with wide white eyes and open mouth, shouting in panic, lunging toward the camera. On the right, a glossy translucent pink butterfly with curled antennae, sharp focus. Shallow depth of field, close-up.",
    "Flat graphic title card on a plain white background. Centered bold cyan-blue gradient sans-serif uppercase text \"WWW.BIGBUCKBUNNY.ORG\". Below it, smaller black condensed uppercase text \"LICENCED AS CREATIVE COMMONS 3.0 ATTRIBUTION\" with a black circular CC icon at left. Minimalist, even lighting, wide margins.",
    "Flat graphic title card on a pure white background. Centered bold cyan-blue gradient uppercase text \"WWW.BIGBUCKBUNNY.ORG\" in wide sans-serif. Below it, a black Creative Commons \"cc\" circle icon beside condensed black uppercase text \"LICENCED AS CREATIVE COMMONS 3.0 ATTRIBUTION\". Clean, minimal, evenly lit. No characters."
]


def test_trailer_title_cards_are_not_data():
    fp = analyze_stylistic_fingerprint(
        _make_shots(TRAILER_DESCS, 1.2), [], 33.0, reading_level="Pre-K to Grade 2")
    assert fp["world_type"] == "Stylized Real-World"
    assert fp["narrative_structure"] != "Data-Driven / Analytical"
    assert fp["tonal_positioning"] == "Child-Friendly / Whimsical"
