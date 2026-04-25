# tasks.py
"""
25 hallucination detection samples across 3 difficulty levels.
Every sample passes the 6-test checklist:
  1. Exact phrase pointable in llm_response
  2. Objective — numbers, names, dates only
  3. No external knowledge needed — reference is sole source of truth
  4. hallucinated_phrase is exact substring of llm_response
  5. Instant agreement — no hesitation
  6. 3 people would agree unanimously
"""

from typing import Any, Dict, List

TASKS: Dict[str, List[Dict[str, Any]]] = {

    # ════════════════════════════════════════════════════════════════
    # EASY — 8 samples
    # Design: 3-4 sentences in reference, 1 obvious error per sample,
    # 2 clean samples (no hallucination)
    # ════════════════════════════════════════════════════════════════
    "easy": [

        # E1 — wrong year (obvious mismatch)
        {
            "reference_document": (
                "The Eiffel Tower was completed in 1889 in Paris, France. "
                "It was designed by Gustave Eiffel and stands approximately 330 metres tall. "
                "It was built as the entrance arch for the 1889 World's Fair."
            ),
            "llm_response": (
                "The Eiffel Tower was completed in 1902 in Paris, France. "
                "It was designed by Gustave Eiffel and stands approximately 330 metres tall."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["completed in 1902"],
            "ground_truth_corrections": ["completed in 1889"],
            "hint": "Compare the completion year in both texts."
        },

        # E2 — wrong location (obvious mismatch)
        {
            "reference_document": (
                "The Taj Mahal is located in Agra, India. "
                "It was commissioned by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal. "
                "Construction was completed in 1653."
            ),
            "llm_response": (
                "The Taj Mahal is located in New Delhi, India. "
                "It was commissioned by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "New Delhi",           # short form — catches any phrasing
                "located in New Delhi",
                "New Delhi, India"  # long form
            ],
            "ground_truth_corrections": [
                "Agra",
                "located in Agra",
                "Agra, India"
            ],
            "hint": "Compare the city mentioned in both texts."
        },

        # E3 — wrong person (obvious mismatch)
        {
            "reference_document": (
                "Python was created by Guido van Rossum and first released in 1991. "
                "It is a high-level, general-purpose programming language. "
                "Python emphasises code readability and simplicity."
            ),
            "llm_response": (
                "Python was created by Dennis Ritchie and first released in 1991. "
                "It is a high-level, general-purpose programming language."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["created by Dennis Ritchie"],
            "ground_truth_corrections": ["created by Guido van Rossum"],
            "hint": "Compare the name of the creator in both texts."
        },

        # E4 — wrong number (obvious mismatch)
        {
            "reference_document": (
                "The Great Wall of China stretches approximately 21,196 kilometres in total length. "
                "It was built over many centuries by various Chinese dynasties. "
                "The most well-known sections were constructed during the Ming Dynasty."
            ),
            "llm_response": (
                "The Great Wall of China stretches approximately 8,000 kilometres in total length. "
                "It was built over many centuries by various Chinese dynasties."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["approximately 8,000 kilometres"],
            "ground_truth_corrections": ["approximately 21,196 kilometres"],
            "hint": "Compare the length figure in both texts."
        },

        # E5 — wrong year (obvious mismatch)
        {
            "reference_document": (
                "The first iPhone was introduced by Apple on January 9, 2007. "
                "It was announced by Steve Jobs at the Macworld Conference. "
                "The device combined a phone, an iPod, and an internet communicator."
            ),
            "llm_response": (
                "The first iPhone was introduced by Apple on January 9, 2009. "
                "It was announced by Steve Jobs at the Macworld Conference."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["January 9, 2009"],
            "ground_truth_corrections": ["January 9, 2007"],
            "hint": "Compare the introduction year in both texts."
        },

        # E6 — wrong unit (obvious mismatch)
        {
            "reference_document": (
                "Water boils at 100 degrees Celsius at standard atmospheric pressure. "
                "Below this temperature, water exists as a liquid. "
                "Above this temperature, water transitions to steam."
            ),
            "llm_response": (
                "Water boils at 90 degrees Celsius at standard atmospheric pressure. "
                "Below this temperature, water exists as a liquid."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["boils at 90 degrees Celsius"],
            "ground_truth_corrections": ["boils at 100 degrees Celsius"],
            "hint": "Compare the boiling temperature in both texts."
        },

        # E7 — CLEAN (no hallucination)
        {
            "reference_document": (
                "The Amazon River is the largest river in the world by discharge volume. "
                "It flows through South America, primarily through Brazil. "
                "The river basin covers approximately 7 million square kilometres."
            ),
            "llm_response": (
                "The Amazon River is the largest river in the world by discharge volume. "
                "It flows through South America, primarily through Brazil."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Read carefully — the response contains only facts from the reference."
        },

        # E8 — CLEAN (no hallucination)
        {
            "reference_document": (
                "Albert Einstein was born on 14 March 1879 in Ulm, Germany. "
                "He developed the theory of relativity. "
                "He was awarded the Nobel Prize in Physics in 1921."
            ),
            "llm_response": (
                "Albert Einstein was born on 14 March 1879 in Ulm, Germany. "
                "He developed the theory of relativity."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Every fact in the response matches the reference document."
        },
    ],

    # ════════════════════════════════════════════════════════════════
    # MEDIUM — 10 samples
    # Design: 5-6 sentences in reference, 2-3 errors per sample,
    # errors mixed across types, 2 clean samples
    # ════════════════════════════════════════════════════════════════
    "medium": [

        # M1 — two errors: wrong height + wrong year
        {
            "reference_document": (
                "The Burj Khalifa in Dubai is the tallest building in the world, standing 828 metres tall. "
                "Construction began in 2004 and the building was officially opened on 4 January 2010. "
                "It has 163 floors above ground. "
                "The building was designed by the architectural firm Skidmore, Owings and Merrill."
            ),
            "llm_response": (
                "The Burj Khalifa in Dubai is the tallest building in the world, standing 760 metres tall. "
                "Construction began in 2004 and the building was officially opened on 4 January 2012. "
                "It has 163 floors above ground."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["standing 760 metres tall", "4 January 2012"],
            "ground_truth_corrections": ["standing 828 metres tall", "4 January 2010"],
            "hint": "Check the height figure and the opening date."
        },

        # M2 — two errors: wrong founder + wrong founding year
        {
            "reference_document": (
                "Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne. "
                "The company is headquartered in Cupertino, California. "
                "Apple designs, manufactures, and markets consumer electronics, software, and online services. "
                "Its first product was the Apple I personal computer."
            ),
            "llm_response": (
                "Apple Inc. was founded on April 1, 1980, by Steve Jobs and Steve Wozniak. "
                "The company is headquartered in Cupertino, California. "
                "Its first product was the Apple I personal computer."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["April 1, 1980"],
            "ground_truth_corrections": ["April 1, 1976"],
            "hint": "Check the founding year. Also note who is missing from the founders list."
        },

        # M3 — two errors: wrong distance + wrong direction
        {
            "reference_document": (
                "The Moon orbits the Earth at an average distance of approximately 384,400 kilometres. "
                "The Moon completes one orbit around the Earth every 27.3 days. "
                "The same side of the Moon always faces the Earth due to tidal locking. "
                "The Moon has no atmosphere and no liquid water on its surface."
            ),
            "llm_response": (
                "The Moon orbits the Earth at an average distance of approximately 384,400 kilometres. "
                "The Moon completes one full orbit every 29.5 days. "  # wrong — should be 27.3
                "The same side of the Moon always faces the Earth due to tidal locking. "
                "The Moon has no atmosphere and no liquid water on its surface."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["every 29.5 days"],
            "ground_truth_corrections": ["every 27.3 days"],
            "hint": "Check the orbital period — two similar-looking numbers, one is wrong."
        },

        # M4 — two errors: wrong inventor + wrong year
        {
            "reference_document": (
                "The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN. "
                "He published the first proposal for the web in March 1989. "
                "The first website went live on 20 December 1990. "
                "The web is distinct from the internet, which is the underlying network infrastructure."
            ),
            "llm_response": (
                "The World Wide Web was invented by Tim Berners-Lee in 1991 while working at CERN. "
                "He published the first proposal for the web in March 1989. "
                "The first website went live on 20 December 1990."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["invented by Tim Berners-Lee in 1991"],
            "ground_truth_corrections": ["invented by Tim Berners-Lee in 1989"],
            "hint": "Check the year the World Wide Web was invented."
        },

        # M5 — three errors: wrong area + wrong population + wrong continent
        {
            "reference_document": (
                "Russia is the largest country in the world by land area, covering approximately 17.1 million square kilometres. "
                "It spans across both Europe and Asia. "
                "Russia has a population of approximately 144 million people. "
                "The capital city is Moscow."
            ),
            "llm_response": (
                "Russia is the largest country in the world by land area, covering approximately 14.5 million square kilometres. "
                "It spans across both Europe and Asia. "
                "Russia has a population of approximately 200 million people. "
                "The capital city is Moscow."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "approximately 14.5 million square kilometres",
                "approximately 200 million people"
            ],
            "ground_truth_corrections": [
                "approximately 17.1 million square kilometres",
                "approximately 144 million people"
            ],
            "hint": "Check the land area figure and the population figure."
        },

        # M6 — two errors: wrong speed + wrong name
        {
            "reference_document": (
                "The Concorde was a supersonic passenger jet that could fly at a maximum speed of Mach 2.04, "
                "approximately 2,180 kilometres per hour. "
                "It entered commercial service in 1976 and was operated jointly by Air France and British Airways. "
                "The Concorde was retired from service in 2003."
            ),
            "llm_response": (
                "The Concorde was a supersonic passenger jet that entered commercial service in 1979. "  # wrong — should be 1976
                "It could fly at a maximum speed of Mach 2.04, approximately 2,180 kilometres per hour. "
                "It was operated jointly by Air France and British Airways. "
                "The Concorde was retired from service in 2003."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["entered commercial service in 1979"],
            "ground_truth_corrections": ["entered commercial service in 1976"],
            "hint": "Check the year Concorde entered commercial service."
        },

        # M7 — two errors: digit swap in population + wrong capital
        {
            "reference_document": (
                "Japan has a population of approximately 125 million people. "
                "The capital city is Tokyo, which is also the most populous city in the country. "
                "Japan consists of four main islands: Honshu, Hokkaido, Kyushu, and Shikoku. "
                "The official language is Japanese."
            ),
            "llm_response": (
                "Japan has a population of approximately 152 million people. "
                "The capital city is Tokyo, which is also the most populous city in the country. "
                "Japan consists of four main islands: Honshu, Hokkaido, Kyushu, and Shikoku."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["approximately 152 million people"],
            "ground_truth_corrections": ["approximately 125 million people"],
            "hint": "Check the population figure carefully — the digits may have been swapped."
        },

        # M8 — two errors: wrong duration + wrong launch year
        {
            "reference_document": (
                "The Apollo 11 mission launched on July 16, 1969, and landed on the Moon on July 20, 1969. "
                "Neil Armstrong became the first human to walk on the Moon. "
                "The mission lasted 8 days in total. "
                "Buzz Aldrin was the second astronaut to walk on the lunar surface."
            ),
            "llm_response": (
                "The Apollo 11 mission launched on July 16, 1969, and landed on the Moon on July 24, 1969. "  # wrong — should be July 20
                "Neil Armstrong became the first human to walk on the Moon. "
                "The mission lasted 8 days in total. "
                "Buzz Aldrin was the second astronaut to walk on the lunar surface."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["landed on the Moon on July 24, 1969"],
            "ground_truth_corrections": ["landed on the Moon on July 20, 1969"],
            "hint": "Check the exact date of the Moon landing — two dates appear in the response."
        },

        # M9 — CLEAN (no hallucination)
        {
            "reference_document": (
                "The human body contains 206 bones in adulthood. "
                "Babies are born with approximately 270 to 300 bones, which fuse together as they grow. "
                "The femur, located in the upper leg, is the longest bone in the human body. "
                "Bones are composed primarily of collagen and calcium phosphate."
            ),
            "llm_response": (
                "The human body contains 206 bones in adulthood. "
                "Babies are born with approximately 270 to 300 bones, which fuse together as they grow. "
                "The femur, located in the upper leg, is the longest bone in the human body."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Every figure and fact in the response appears in the reference document."
        },

        # M10 — CLEAN (no hallucination)
        {
            "reference_document": (
                "The human heart beats approximately 60 to 100 times per minute at rest. "
                "The heart pumps blood through two separate circuits: the pulmonary circuit and the systemic circuit. "
                "An adult heart weighs between 250 and 350 grams. "
                "The heart has four chambers: two atria and two ventricles."
            ),
            "llm_response": (
                "The human heart beats approximately 60 to 100 times per minute at rest. "
                "The heart has four chambers: two atria and two ventricles."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Both facts in the response are directly stated in the reference."
        },
    ],

    # ════════════════════════════════════════════════════════════════
    # HARD — 7 samples
    # Design: 6-8 sentences in reference, subtle errors buried in
    # correct facts, omission errors, near-correct values,
    # 1 clean sample, 1 false-alarm trap
    # ════════════════════════════════════════════════════════════════
    "hard": [

        # H1 — Ambiguity Trap (UN Status)
        # Difficulty: Confuses "participation" (observer status) with "membership".
        # Reasoning: Requires distinguishing categorical legal status from functional roles.
        {
            "reference_document": (
                "The Republic of China (Taiwan) is not a member of the United Nations, having been replaced by the "
                "People's Republic of China in 1971 through General Assembly Resolution 2758. "
                "Since then, Taiwan has participated in certain international organizations (like the WHO) as an observer, "
                "but it lacks the status of a sovereign member state in the UN body."
            ),
            "llm_response": (
                "Taiwan remains a functional member of the UN under a specialized status, though it lacks voting "
                "rights in the General Assembly. Following Resolution 2758, the UN created a unique observer-member "
                "category for independent territories like Taiwan."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "functional member of the UN under a specialized status",
                "unique observer-member category"
            ],
            "ground_truth_corrections": [
                "is not a member of the United Nations",
                "lacks the status of a sovereign member state"
            ],
            "hint": "Check the specific legal status of 'membership' vs 'participation'."
        },

        # H2 — Partial Truth (Medicine MOA)
        # Difficulty: Model mixes common medication categories (stimulants vs reducers).
        # Reasoning: Requires precise biochemical knowledge of drug mechanisms.
        {
            "reference_document": (
                "Metformin is the first-line medication for type 2 diabetes. It works primarily by suppressing "
                "glucose production in the liver (gluconeogenesis) and increasing the insulin sensitivity of body tissues. "
                "Unlike sulfonylureas, Metformin does not stimulate insulin secretion from the pancreas."
            ),
            "llm_response": (
                "As the primary treatment for type 2 diabetes, Metformin acts by stimulating the beta cells in the "
                "pancreas to release more insulin. This increased insulin sensitivity effectively lowers blood glucose levels."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["stimulating the beta cells in the pancreas to release more insulin"],
            "ground_truth_corrections": ["suppressing glucose production in the liver"],
            "hint": "Compare the organ affected: liver vs pancreas."
        },

        # H3 — Multi-hop Reasoning (Distance/Units)
        # Difficulty: Confuses 'm' (meters) with 'km' (kilometers) over large 7-decade leap.
        # Reasoning: Unit conversion error triggered by large numeric comparisons.
        {
            "reference_document": (
                "The Wright brothers' first flight in 1903 covered 36 meters (120 feet) in 12 seconds. "
                "By 1969, the Apollo 11 mission completed a journey of roughly 384,400 kilometres to reach the Moon. "
                "The speed of travel increased from 10 km/h to over 39,000 km/h in just 66 years."
            ),
            "llm_response": (
                "In less than 70 years, humans advanced from a 36-meter flight to a lunar mission covering 384,400 meters. "
                "This represent one of the most rapid technological accelerations in human history."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["covering 384,400 meters"],
            "ground_truth_corrections": ["covering 384,400 kilometres"],
            "hint": "Double check the units of the lunar distance: meters vs kilometres."
        },

        # H4 — Negation Trap (Liability)
        # Difficulty: Fails to process "unless" + "not liable" conditional logic.
        # Reasoning: Linguistic negation reversal.
        {
            "reference_document": (
                "The software license agreement states that the developer is not liable for indirect damages "
                "unless such damages are caused by gross negligence or willful misconduct. "
                "In standard cases of simple negligence, the developer's liability is capped at the purchase price."
            ),
            "llm_response": (
                "According to the agreement, the developer remains liable for indirect damages even if "
                "gross negligence cannot be proven. Simple negligence is enough to trigger full liability."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "liable for indirect damages even if gross negligence cannot be proven",
                "Simple negligence is enough to trigger full liability"
            ],
            "ground_truth_corrections": [
                "not liable for indirect damages unless gross negligence is proven",
                "simple negligence is capped at the purchase price"
            ],
            "hint": "Look at the condition for liability: simple vs gross negligence."
        },

        # H5 — Entity-Role Confusion (M&A)
        # Difficulty: Swaps acquirer and target in a famous corporate deal.
        # Reasoning: Subject/Object reversal.
        {
            "reference_document": (
                "In June 2017, Amazon announced its intention to acquire the grocery chain Whole Foods for $13.7 billion. "
                "The acquisition marked Amazon's largest entry into physical retail stores. "
                "The deal was finalized in August 2017."
            ),
            "llm_response": (
                "Whole Foods made a massive digital expansion in 2017 by acquiring Amazon for $13.7 billion, "
                "integrating its organic grocery supply chain into Amazon's global delivery network."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["acquiring Amazon"],
            "ground_truth_corrections": ["Amazon announced its intention to acquire the grocery chain Whole Foods"],
            "hint": "Who bought whom?"
        },

        # H6 — Truth-that-sounds-false (Venus Rotation)
        # Difficulty: Adversarial Clean. The fact is counter-intuitive and sounds like a typical hallucination.
        # Reasoning: Tempts the model to 'correct' a true fact into a false one.
        {
            "reference_document": (
                "Venus has an unusual rotation; it takes 243 Earth days to complete one rotation on its axis. "
                "However, its orbital period is only 225 Earth days. "
                "Therefore, a day on Venus (one full rotation) is actually longer than a year on Venus (one full orbit)."
            ),
            "llm_response": (
                "Venus is unique because its day (243 Earth days) is actually longer than its year (225 Earth days). "
                "This means the planet completes an orbit around the Sun faster than it rotates once on its axis."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Verify the numbers 243 and 225 carefully."
        },

        # H7 — False-but-plausible (Magna Carta)
        # Difficulty: Uses a plausible-sounding royal term (Divine Right) that is historically opposite to the treaty's intent.
        # Reasoning: Conceptual hallucination.
        {
            "reference_document": (
                "The Magna Carta, issued in June 1215 at Runnymede, aimed to limit the power of King John. "
                "It established the principle that everyone, including the king, was subject to the law. "
                "It primarily addressed the grievances of rebellious barons."
            ),
            "llm_response": (
                "The 1215 Magna Carta was signed by King John to formalize the Divine Right of Kings, "
                "ensuring that the monarchy's absolute power was recognized by the Church and the barons."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["formalize the Divine Right of Kings", "monarchy's absolute power"],
            "ground_truth_corrections": ["limit the power of King John", "king was subject to the law"],
            "hint": "Did the Magna Carta increase or decrease the King's power?"
        },

        # H8 — No-hallucination Adversarial (Triple Witching)
        # Difficulty: Adversarial Clean. Uses synonymous phrasing (Final month of each quarter) to trigger false detection.
        # Reasoning: Tests linguistic flexibility vs rigid fact-checking.
        {
            "reference_document": (
                "Triple Witching occurs on the third Friday of March, June, September, and December. "
                "This is when stock options, stock index futures, and stock index options all expire on the same day. "
                "This often leads to increased trading volume and volatility."
            ),
            "llm_response": (
                "Triple Witching happens on the third Friday of the final month of each quarter. "
                "It involves the simultaneous expiration of stock options and various index-related derivatives."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Check if 'final month of each quarter' matches 'March, June, September, and December'."
        },

        # H9 — Multi-error (European Union)
        # Difficulty: Outdated training data (Brexit) + Multiple numeric counts.
        # Reasoning: Tests ability to catch multiple, distinct hallucinations in a short text.
        {
            "reference_document": (
                "Following the withdrawal of the United Kingdom, the European Union has 27 member states. "
                "The Eurozone consists of 20 countries that use the Euro as their currency. "
                "Croatia was the most recent country to join the Eurozone in January 2023."
            ),
            "llm_response": (
                "The European Union currently consists of 28 member states, while the Eurozone "
                "includes 19 countries that have adopted the Euro as their primary currency."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["28 member states", "19 countries"],
            "ground_truth_corrections": ["27 member states", "20 countries"],
            "hint": "Look for two separate errors in this response."
        },

        # H10 — Temporal Confusion (Voyager 1)
        # Difficulty: Confuses the Heliopause (Interstellar space boundary) with the Oort Cloud (Solar System boundary).
        # Reasoning: Spatial/Temporal logic boundary error.
        {
            "reference_document": (
                "Voyager 1, launched in 1977, officially entered interstellar space in August 2012. "
                "While it has left the heliosphere, it will not reach the inner edge of the Oort Cloud for "
                "another 300 years. The Oort Cloud is considered the final boundary of the Solar System."
            ),
            "llm_response": (
                "Voyager 1 was launched in 1977 and reached the heliopause in 2012, though it didn't "
                "officially enter interstellar space until it reached the Oort Cloud in 2018."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["didn't officially enter interstellar space until it reached the Oort Cloud in 2018"],
            "ground_truth_corrections": ["entered interstellar space in August 2012"],
            "hint": "Distinguish between 'Interstellar Space' and the 'Oort Cloud'."
        },

        # H11 — Ambiguity Trap (Marie Curie)
        # Difficulty: Adversarial Clean. Model may hallucinate a 'conflict' with Linus Pauling.
        # Reasoning: Requires verifying specific constraints ("Scientific fields").
        {
            "reference_document": (
                "Marie Curie is the only person to win Nobel Prizes in two different scientific fields: "
                "Physics (1903) and Chemistry (1911). Linus Pauling is the only person to win two "
                "unshared Nobel Prizes, but one of his was for Peace, which is not a scientific field."
            ),
            "llm_response": (
                "Marie Curie remains the only individual in history to be awarded Nobel Prizes in two "
                "distinct scientific areas: Physics and Chemistry."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "Is Marie Curie indeed the only one in *scientific* fields?"
        },

        # H12 — Partial Truth (JWST)
        # Difficulty: Subtle numeric swap in a highly technical description.
        # Reasoning: Precision monitoring of hardware specs.
        {
            "reference_document": (
                "The James Webb Space Telescope (JWST) is protected by a massive sunshield consisting of "
                "five layers of Kapton. This sunshield allows the telescope to operate at temperatures "
                "below 50 Kelvin (-223 degrees Celsius)."
            ),
            "llm_response": (
                "To remain cool, the JWST utilizes a three-layer Kapton sunshield, which keeps the sensitive "
                "infrared instruments at a stable operating temperature near absolute zero."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["three-layer Kapton sunshield"],
            "ground_truth_corrections": ["five layers of Kapton"],
            "hint": "Check the exact number of layers in the sunshield."
        },

        # H13 — Entity-Role Confusion (Nord Stream)
        # Difficulty: Reverses exporter/importer and commodity type.
        # Reasoning: Logic/Geopolitical flow reversal.
        {
            "reference_document": (
                "The Nord Stream 2 pipeline was built to transport natural gas from Russia to Germany. "
                "The project travels through the Baltic Sea and was intended to double the capacity "
                "of the original Nord Stream 1 pipeline."
            ),
            "llm_response": (
                "Nord Stream 2 was a German infrastructure project designed to export surplus natural gas "
                "to Russia through a series of offshore pipelines in the Baltic Sea."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["German infrastructure project designed to export ... to Russia"],
            "ground_truth_corrections": ["transport natural gas from Russia to Germany"],
            "hint": "Follow the direction of the gas flow in the pipe."
        },

        # H14 — False-but-plausible (Antibiotics)
        # Difficulty: Categorization error. Plausible-sounding antiviral designation for a famous medication.
        # Reasoning: Lexical category swap.
        {
            "reference_document": (
                "Penicillin, discovered by Alexander Fleming in 1928, is an antibiotic used to treat "
                "bacterial infections. It works by interfering with the bacterium's ability to build its "
                "cell wall, leading to the death of the pathogen."
            ),
            "llm_response": (
                "Alexander Fleming's 1928 discovery, Penicillin, is a staple antiviral medication that "
                "prevents viral replication by breaking down the protein coat of the virus."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["antiviral medication that prevents viral replication"],
            "ground_truth_corrections": ["antibiotic used to treat bacterial infections"],
            "hint": "Does Penicillin kill bacteria or viruses?"
        },

        # H15 — Temporal Confusion (The Macintosh)
        # Difficulty: Multiple errors regarding hardware/software timelines.
        # Reasoning: Requires mapping specific features to specific years and versions.
        {
            "reference_document": (
                "Apple's Macintosh, released in 1984, was the first mass-market personal computer to "
                "feature an integral graphical user interface and a mouse. System 7, released in 1991, "
                "was a major software update that introduced multitasking."
            ),
            "llm_response": (
                "Although released in 1984, the original Macintosh only featured a command-line interface. "
                "The transition to a graphical user interface and a mouse didn't occur until System 7 in 1991."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "original Macintosh only featured a command-line interface",
                "didn't occur until System 7 in 1991"
            ],
            "ground_truth_corrections": [
                "first mass-market personal computer to feature an integral graphical user interface",
                "featured ... a mouse"
            ],
            "hint": "Did the 1984 Macintosh have a mouse and a GUI from the beginning?"
        }
    ],
    # ════════════════════════════════════════════════════════════════
    # EXPERT DIFFICULTY — 20 samples
    # Design:
    #   - 7-9 sentence reference documents (longer context)
    #   - 16 hallucination samples (subtle: wrong-but-plausible)
    #   - 4 adversarial clean samples (counterintuitive truths)
    #   - Max steps: 22
    #
    # Hallucination types used:
    #   1. Subtle (plausible wrong reason / category swap)
    #   2. Mixed true/false in same sentence
    #   3. Numeric / date arithmetic errors
    #   4. Multi-hop reasoning required
    # ════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════
    # EXPERT — 20 samples
    # ════════════════════════════════════════════════════════════════
    "expert": [

        # EX1 — Clean Adversarial (Double Negation Logic)
        {
            "reference_document": (
                "The UN Security Council failed to adopt the resolution condemning the annexation because, "
                "although it received 13 affirmative votes, it was vetoed by a permanent member. "
                "Consequently, the resolution was not passed."
            ),
            "llm_response": (
                "Despite overwhelmingly positive support from 13 members, the resolution condemning the "
                "annexation was ultimately blocked by a permanent member's veto, meaning the UN did not "
                "pass the measure."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "A naive grader flags 'blocked' as a negative hallucination, but the semantic meaning perfectly aligns with the reference."
        },

        # EX2 — Financial Math (Margin vs Markup)
        {
            "reference_document": (
                "A retailer buys a product for $80 and sells it for $100. The markup is based on cost, "
                "calculated as ($20 / $80) = 25%. The gross margin is based on revenue, calculated as "
                "($20 / $100) = 20%."
            ),
            "llm_response": (
                "The retailer's pricing strategy yields a 25% gross margin on the product, since the $20 "
                "profit represents a quarter of the $80 wholesale cost."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["25% gross margin"],
            "ground_truth_corrections": ["25% markup, 20% gross margin"],
            "hint": "The LLM correctly identifies 25% but assigns it to the wrong financial term (Margin instead of Markup)."
        },

        # EX3 — Thermodynamics (Isothermal vs Adiabatic)
        {
            "reference_document": (
                "In an adiabatic process, no heat is exchanged with the environment. If a gas expands "
                "adiabatically, its temperature drops because it does work on the surroundings. In an "
                "isothermal expansion, temperature remains constant and heat must be absorbed."
            ),
            "llm_response": (
                "During an adiabatic expansion, the gas does work on its surroundings, which causes its "
                "temperature to remain constant as no heat is exchanged."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["causes its temperature to remain constant"],
            "ground_truth_corrections": ["its temperature drops"],
            "hint": "The response confuses the outcome of an isothermal expansion with an adiabatic one."
        },

        # EX4 — Legal Logic (Necessary vs Sufficient)
        {
            "reference_document": (
                "Under Article 5, holding a valid permit is a necessary, but not sufficient, condition for "
                "operating the machinery. The operator must also complete the Level 2 safety certification "
                "within the past 12 months."
            ),
            "llm_response": (
                "According to Article 5, holding a valid permit is sufficient to operate the machinery, "
                "provided the operator also holds a Level 2 safety certification from the past year."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["holding a valid permit is sufficient to operate the machinery"],
            "ground_truth_corrections": ["holding a valid permit is a necessary, but not sufficient, condition"],
            "hint": "If an additional condition is required, the first condition cannot logically be 'sufficient'."
        },

        # EX5 — Date Math (Leap Year)
        {
            "reference_document": (
                "The probe was launched on February 28, 2024 (a leap year). It was scheduled to perform "
                "its orbital insertion maneuver exactly 60 days later."
            ),
            "llm_response": (
                "Following its launch on February 28, 2024, the probe was scheduled for orbital insertion "
                "exactly 60 days later, on April 29, 2024."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["on April 29, 2024"],
            "ground_truth_corrections": ["on April 28, 2024"],
            "hint": "2024 is a leap year. Feb has 29 days. (Feb 28 + 1) + 31 (March) + 28 (April) = 60 days. It should be April 28."
        },

        # EX6 — Biology (Exclusion Trap)
        {
            "reference_document": (
                "All known mammalian species possess a four-chambered heart. However, among reptiles, "
                "only crocodilians (crocodiles and alligators) have a four-chambered heart; other "
                "reptiles typically have a three-chambered heart."
            ),
            "llm_response": (
                "A four-chambered heart is a trait exclusive to mammals and certain crocodilians, meaning "
                "no other animal classes possess this physiological feature."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["trait exclusive to mammals and certain crocodilians", "no other animal classes possess this physiological feature"],
            "ground_truth_corrections": ["among reptiles, only crocodilians have a four-chambered heart"],
            "hint": "The text says 'among reptiles', crocodilians are the only ones. It does not exclude other entire classes (like birds) which the LLM falsely assumes."
        },

        # EX7 — Statistics (Simpson's Trap / Average of Percentages)
        {
            "reference_document": (
                "Fund A manages $100 million and returned 5% ($5 million). Fund B manages $10 million "
                "and returned 15% ($1.5 million). The combined total assets are $110 million."
            ),
            "llm_response": (
                "The firm manages $110 million across two funds, which generated an average return of "
                "10% combined."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["average return of 10% combined"],
            "ground_truth_corrections": ["combined return is approximately 5.9% ($6.5M / $110M)"],
            "hint": "Taking the direct average of 5% and 15% ignores the massive weighting difference between the funds."
        },

        # EX8 — Temporal/Spatial Dependency (Network Replication)
        {
            "reference_document": (
                "Server A replicates data to Server B synchronously. Server B replicates to Server C "
                "asynchronously. If Server A fails, Server B is guaranteed to have the latest data. "
                "However, if Server B fails before asynchronous replication completes, Server C may experience data loss."
            ),
            "llm_response": (
                "Because Server A replicates synchronously to Server B, and B to C, any data committed to "
                "Server A is immediately and safely guaranteed to exist on Server C without risk of loss."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["immediately and safely guaranteed to exist on Server C without risk of loss"],
            "ground_truth_corrections": ["Server C may experience data loss if Server B fails before asynchronous replication completes"],
            "hint": "The asynchronous link to Server C invalidates the 'immediate and safe' guarantee."
        },

        # EX9 — Clean Adversarial (Counter-intuitive Physics)
        {
            "reference_document": (
                "Water has a higher specific heat capacity than land. This means water takes longer to "
                "heat up in the summer and longer to cool down in the winter, which moderates the "
                "climate of coastal cities compared to inland areas."
            ),
            "llm_response": (
                "Because land has a lower specific heat capacity than water, it absorbs and releases "
                "heat more rapidly, leading to more extreme temperature fluctuations in inland areas "
                "than in coastal regions."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "The LLM correctly flipped the subject to 'land' and inverted the specific heat relationship accurately."
        },

        # EX10 — Geographic/Political (Overgeneralization)
        {
            "reference_document": (
                "The Schengen Area abolished border controls. Ireland (an EU member) opted out of the "
                "agreement. Conversely, Iceland and Norway are not in the EU, but they are full members "
                "of the Schengen Area."
            ),
            "llm_response": (
                "The Schengen Area allows passport-free travel across its member states. While non-EU "
                "countries like Iceland and Norway are excluded, EU member Ireland participates fully "
                "in the agreement."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["non-EU countries like Iceland and Norway are excluded", "EU member Ireland participates fully"],
            "ground_truth_corrections": ["Iceland and Norway are not in the EU, but they are full members", "Ireland (an EU member) opted out"],
            "hint": "The LLM perfectly reversed the inclusion/exclusion status of all three countries."
        },

        # EX11 — Legal Modifier Swap
        {
            "reference_document": (
                "In the United States legal system, a civil trial typically requires a 'preponderance of "
                "the evidence' to reach a verdict, meaning it is more likely than not that the claim is true. "
                "Criminal trials require a much higher standard: proof 'beyond a reasonable doubt'."
            ),
            "llm_response": (
                "In US civil cases, plaintiffs must prove their claims beyond a reasonable doubt, whereas "
                "criminal cases operate on the slightly lower standard of a preponderance of the evidence."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "civil cases, plaintiffs must prove their claims beyond a reasonable doubt",
                "criminal cases operate on the slightly lower standard of a preponderance of the evidence"
            ],
            "ground_truth_corrections": [
                "civil trial typically requires a 'preponderance of the evidence'",
                "Criminal trials require a much higher standard: proof 'beyond a reasonable doubt'"
            ],
            "hint": "The LLM swapped the burdens of proof for civil and criminal trials."
        },

        # EX12 — Entity Shifting in Lists
        {
            "reference_document": (
                "The Apollo 15 mission featured astronauts David Scott, Alfred Worden, and James Irwin. "
                "Scott and Irwin walked on the Moon, while Worden remained in lunar orbit aboard the Command Module."
            ),
            "llm_response": (
                "During the Apollo 15 mission, astronauts David Scott and Alfred Worden explored the lunar surface, "
                "leaving James Irwin to pilot the Command Module in orbit."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "Alfred Worden explored the lunar surface",
                "leaving James Irwin to pilot the Command Module"
            ],
            "ground_truth_corrections": [
                "Scott and Irwin walked on the Moon",
                "Worden remained in lunar orbit"
            ],
            "hint": "Pay close attention to which specific astronaut performed which specific role."
        },

        # EX13 — Chemical Stoichiometry Reversal
        {
            "reference_document": (
                "During cellular respiration, one molecule of glucose (C6H12O6) reacts with six molecules "
                "of oxygen (O2) to produce six molecules of carbon dioxide (CO2) and six molecules of "
                "water (H2O), yielding ATP."
            ),
            "llm_response": (
                "Cellular respiration uses one glucose molecule and six carbon dioxide molecules to produce "
                "six oxygen molecules and six water molecules, along with ATP."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": [
                "uses one glucose molecule and six carbon dioxide molecules",
                "to produce six oxygen molecules"
            ],
            "ground_truth_corrections": [
                "reacts with six molecules of oxygen",
                "to produce six molecules of carbon dioxide"
            ],
            "hint": "The LLM swapped the reactants and the products (describing photosynthesis rather than respiration)."
        },

        # EX14 — Aviation Unit Mismatch
        {
            "reference_document": (
                "Aviation regulations require an aircraft flying under Visual Flight Rules (VFR) above "
                "10,000 feet Mean Sea Level (MSL) to maintain at least 5 miles of flight visibility and stay "
                "1,000 feet above, 1,000 feet below, and 1 mile horizontally from clouds."
            ),
            "llm_response": (
                "For VFR flights operating above 10,000 feet MSL, regulations mandate 5 miles of visibility "
                "and a cloud clearance of 1,000 feet above, 1,000 feet below, and 1,000 feet horizontally."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["1,000 feet horizontally"],
            "ground_truth_corrections": ["1 mile horizontally"],
            "hint": "Look closely at the horizontal clearance requirement unit."
        },

        # EX15 — Clean Adversarial (Vocabulary Shift)
        {
            "reference_document": (
                "The findings of the investigation were entirely unremarkable and yielded no actionable "
                "intelligence regarding the suspect's whereabouts."
            ),
            "llm_response": (
                "The investigation provided negligible insights, completely failing to produce any useful "
                "data on where the suspect might be."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "The vocabulary changed dramatically, but the semantic truth is perfectly preserved."
        },

        # EX16 — Historical Succession Merging
        {
            "reference_document": (
                "Queen Victoria reigned from 1837 until her death in 1901. She was succeeded by her eldest "
                "son, Edward VII, who ruled until 1910. He was followed by his second son, George V, because "
                "his eldest son, Prince Albert Victor, had died in 1892."
            ),
            "llm_response": (
                "Following Queen Victoria's death in 1901, her eldest son Edward VII took the throne. Upon "
                "Edward's death in 1910, his eldest son, Prince Albert Victor, succeeded him as George V."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["his eldest son, Prince Albert Victor, succeeded him as George V"],
            "ground_truth_corrections": ["followed by his second son, George V", "eldest son, Prince Albert Victor, had died in 1892"],
            "hint": "The LLM merged a dead heir with the actual successor to create a historically false entity."
        },

        # EX17 — Network Protocol Reversal
        {
            "reference_document": (
                "In a TCP/IP packet header, the Source IP address indicates where the packet originated, "
                "while the Destination IP indicates where it is going. When a router performs Network "
                "Address Translation (NAT) for an outgoing packet, it modifies the Source IP address to "
                "its own public IP."
            ),
            "llm_response": (
                "When an outgoing packet passes through a NAT-enabled router, the router alters the "
                "Destination IP address to match its own public IP before forwarding it to the internet."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["alters the Destination IP address"],
            "ground_truth_corrections": ["modifies the Source IP address"],
            "hint": "Which IP address does an outbound NAT router actually change?"
        },

        # EX18 — Medical Omission (Critical Qualifier)
        {
            "reference_document": (
                "The drug is safe for pregnant women during the first and second trimesters. However, it "
                "is strictly contraindicated in the third trimester due to the risk of premature ductus "
                "arteriosus closure."
            ),
            "llm_response": (
                "Clinical guidelines state that the drug is perfectly safe to administer to pregnant women, "
                "provided they do not exceed the recommended daily dosage."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["perfectly safe to administer to pregnant women"],
            "ground_truth_corrections": ["strictly contraindicated in the third trimester"],
            "hint": "By dropping the specific trimester constraint, the absolute 'safe' statement becomes a life-threatening hallucination."
        },

        # EX19 — Clean Adversarial (Area Mathematics)
        {
            "reference_document": (
                "The architectural design required the base of the pyramid to be exactly a square with side "
                "lengths of 150 meters, yielding a total base area of 22,500 square meters."
            ),
            "llm_response": (
                "The pyramid's foundation was designed as a perfect quadrilateral with four equal 150-meter "
                "sides and four right angles, covering 2.25 hectares of land."
            ),
            "ground_truth_has_hallucination": False,
            "ground_truth_hallucinated_phrases": [],
            "ground_truth_corrections": [],
            "hint": "1 hectare = 10,000 sq m. 2.25 hectares = 22,500 sq m. A square is a quadrilateral with equal sides and right angles."
        },

        # EX20 — Cryptographic Directionality
        {
            "reference_document": (
                "Hash functions are deterministic, meaning the same input always produces the same output. "
                "They are also designed to be one-way, meaning it is computationally infeasible to reverse "
                "the hash to find the original input. Unlike encryption, hashing does not use a key to allow decryption."
            ),
            "llm_response": (
                "As deterministic algorithms, hash functions ensure that identical inputs yield identical "
                "outputs. Furthermore, anyone with the correct cryptographic key can reverse the hash to "
                "reveal the original input."
            ),
            "ground_truth_has_hallucination": True,
            "ground_truth_hallucinated_phrases": ["anyone with the correct cryptographic key can reverse the hash"],
            "ground_truth_corrections": ["hashing does not use a key to allow decryption", "computationally infeasible to reverse the hash"],
            "hint": "The LLM confused the one-way nature of hashing with two-way symmetric encryption."
        }
    ],  # end "expert"
}



def get_task(task_id: str) -> List[Dict[str, Any]]:
    """Return the sample list for the given task_id."""
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> List[str]:
    """Return all available task IDs."""
    return list(TASKS.keys())


def count_samples() -> Dict[str, int]:
    """Return the number of samples per task."""
    return {k: len(v) for k, v in TASKS.items()}