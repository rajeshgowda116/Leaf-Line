// Generator script to build and validate the Tree Planting Roadmap integration
const fs = require('fs');

const TREE_PLANTING_SCHEDULE = [
    {
        week: 1,
        days: "Days 1–7",
        phase: "Phase 1 — Site Selection, Sunlight & Safe Distance Planning",
        goal: "Choose the right sunny spot, verify soil drainage, and measure safe distances from buildings and pipes.",
        notes: "Trusted Science (ICAR-CAZRI & FAO Tree Planting Manual): Fruit and shade trees require at least 6 to 8 hours of direct sunshine daily. Test soil drainage by filling a 1-foot hole with water; healthy soil drains within 2 to 3 hours.",
        watchFor: [
            "Distance rule: Plant large shade trees at least 15 to 20 feet away from house walls, septic tanks, and paved driveways.",
            "Look up! Check for overhead power wires that may tangle with branches later as the tree matures.",
            "Avoid low waterlogged depressions where puddle water sits for days after rain."
        ],
        items: [
            { day: "Day 1", observation: "Walk the yard to identify open areas that receive full morning and midday sunlight.", task: "Observe sun patterns from 8 AM to 4 PM; mark sunny candidate spots with garden flags." },
            { day: "Day 2", observation: "Measure distance from boundary walls, underground pipes, and overhead cables.", task: "Ensure a minimum 15-foot radius of open space around the future tree canopy." },
            { day: "Day 3", observation: "Dig a 30 cm drainage test hole; fill with water to observe draining speed.", task: "Fill the hole to the top with water and time how fast it drains downward." },
            { day: "Day 4", observation: "Check drainage results.", task: "If the hole emptied in 2–3 hours, drainage is ideal! If water is still standing after 12 hours, choose a higher spot." },
            { day: "Day 5", observation: "Select suitable tree species adapted to your local climate (Mango, Guava, Lemon, Neem, Jamun).", task: "Match tree size to your garden space; consult local horticulture guidelines." },
            { day: "Day 6", observation: "Mark the pit boundary on the ground with chalk powder or wooden pegs (1m × 1m).", task: "Draw a 1-meter square on the ground to guide the digging crew or shovel work." },
            { day: "Day 7", observation: "Gather digging tools (spade, pickaxe, crowbar) and wheelbarrow.", task: "Ensure tools are clean and ready for pit excavation tomorrow." }
        ]
    },
    {
        week: 2,
        days: "Days 8–14",
        phase: "Phase 2 — Digging the 3×3×3 Pit & Sun Solarization",
        goal: "Dig a generous 1m × 1m × 1m pit and let the bright sun naturally sterilize and sanitize the soil.",
        notes: "Trusted Science (TNAU Forestry & ICAR Horticultural Guide): Standard pits (1m × 1m × 1m / 3ft × 3ft × 3ft) provide loose, uncompacted earth for baby roots. Sun exposure ('Solarization') for 10 to 15 days destroys harmful root-rot fungi, termite eggs, and weed roots without chemicals.",
        watchFor: [
            "Two soil piles: Keep the dark, fertile topsoil in one pile and the reddish hard subsoil in another pile.",
            "Break bottom hardpan: If you hit hard rock or chalk at the pit floor, use a crowbar to fracture it so taproots can punch through.",
            "Safety: Cover or mark open pits with bright tape or branches so children and pets stay safe."
        ],
        items: [
            { day: "Day 8", observation: "Begin digging the marked 1m × 1m square pit.", task: "Shovel the first 30 cm of dark topsoil into a clean pile on the left side." },
            { day: "Day 9", observation: "Continue digging the deeper subsoil (30–60 cm depth).", task: "Heap subsoil into a separate second pile on the right side." },
            { day: "Day 10", observation: "Dig to the full 1-meter (3-foot) depth.", task: "Square off the pit corners neatly and shovel out all loose dirt." },
            { day: "Day 11", observation: "Pierce the floor of the pit with a crowbar 15–20 cm deep.", task: "Loosen the compacted pit base so the future taproot can easily grow downward." },
            { day: "Day 12", observation: "Leave the pit completely open to absorb bright, direct sunshine.", task: "Sun heat acts as a natural disinfectant, destroying dormant fungal spores." },
            { day: "Day 13", observation: "Spread the excavated soil heaps thinly on a tarp or ground.", task: "Bake out hidden beetle grubs and weed seeds under the hot sun." },
            { day: "Day 14", observation: "Inspect pit walls; remove any sharp construction debris, cement chunks, or plastic.", task: "Ensure the pit environment is clean, natural, and free of foreign debris." }
        ]
    },
    {
        week: 3,
        days: "Days 15–21",
        phase: "Phase 3 — Soil Enrichment, Neem Cake & Bio-Shield",
        goal: "Blend excavated topsoil with aged manure, neem cake, and biofungicides to build a rich underground banquet for roots.",
        notes: "Trusted Science (ICAR-IIHR Soil Science Division): Mixing 15–20 kg well-rotted farmyard manure with 500g Neem cake powder provides organic nutrition while warding off subterranean termites. Trichoderma viride bio-fungicide safeguards against damping-off and collar rot.",
        watchFor: [
            "Never use fresh, wet, smelly cow dung — it ferments in the soil, generates intense heat, and burns baby roots. Use only cool, dark, earthy-smelling compost.",
            "Thorough mixing: Blend neem cake and compost evenly into the topsoil.",
            "Avoid chemical urea: Chemical fertilizers in planting pits cause salt burns on young roots."
        ],
        items: [
            { day: "Day 15", observation: "Source 15–20 kg of dark, crumbly vermicompost or 1-year-old aged cow manure.", task: "Check manure quality: it must smell like sweet forest soil, not fresh manure." },
            { day: "Day 16", observation: "Measure 500 grams of pure neem seed cake powder.", task: "Neem cake acts as a natural repellent against termites and root-knot nematodes." },
            { day: "Day 17", observation: "Add 50 grams of Trichoderma viride bio-fungicide powder to the compost heap.", task: "Friendly Trichoderma fungi colonize the soil and eat harmful root-rot fungi." },
            { day: "Day 18", observation: "Add 100 grams of bone meal or rock phosphate.", task: "Provides slow-release phosphorus essential for rapid root system branching." },
            { day: "Day 19", observation: "Thoroughly mix the amendments into the excavated topsoil using a garden spade.", task: "Blend until the soil is light, fluffy, and uniformly rich in organic matter." },
            { day: "Day 20", observation: "Shovel the enriched soil mixture back into the pit until it reaches ground level.", task: "Fill the pit completely and mound it 5 cm higher than surrounding ground to allow for settling." },
            { day: "Day 21", observation: "Soak the filled pit lightly with water.", task: "Settles loose soil naturally and closes major air cavities before planting day." }
        ]
    },
    {
        week: 4,
        days: "Days 22–28",
        phase: "Phase 4 — Nursery Sapling Selection & Acclimatization",
        goal: "Select a vigorous, disease-free nursery sapling and help it adjust to outdoor breezes and sunshine.",
        notes: "Trusted Science (FAO Nursery Management Handbook): A healthy sapling is 2 to 3 feet tall with a sturdy, upright stem, vibrant green leaves, and an unbroken root ball. Avoid overgrown, root-bound saplings whose taproots have penetrated through the nursery bag into the ground.",
        watchFor: [
            "Graft union check: If buying a grafted fruit tree (mango, guava, citrus), ensure the graft joint is firmly healed, straight, and at least 6 inches above the polybag soil line.",
            "Check leaf undersides: Reject saplings showing yellow spotting, spider mites, or curled deformed leaves.",
            "Transport with care: Hold the sapling by the root ball container, NEVER by the thin trunk or green stem."
        ],
        items: [
            { day: "Day 22", observation: "Visit an accredited government or certified private horticulture nursery.", task: "Look for saplings with straight leaders and healthy lateral branches." },
            { day: "Day 23", observation: "Inspect candidate saplings: look for strong central leader stems and rich leaf foliage.", task: "Inspect the graft union: it should be clean and smooth with no peeling bark." },
            { day: "Day 24", observation: "Check the bottom of the black polybag.", task: "Avoid bags with thick tangled roots busting out into ground soil; choose well-contained root balls." },
            { day: "Day 25", observation: "Transport the sapling home in a sheltered vehicle.", task: "Protect green leaves from wind whipping during transport; never let saplings fly in open truck beds." },
            { day: "Day 26", observation: "Place the sapling in a cool, semi-shaded spot outdoors for 2 to 3 days.", task: "Allows the sapling to acclimatize to local wind, humidity, and temperature." },
            { day: "Day 27", observation: "Water the polybag thoroughly the evening before planting.", task: "Ensures the soil ball stays cohesive and roots remain hydrated during transplanting." },
            { day: "Day 28", observation: "Prepare planting accessories: sharp blade, garden hose, 6-foot bamboo stake, and flexible twine.", task: "Gather all gear so planting day proceeds smoothly and quickly." }
        ]
    },
    {
        week: 5,
        days: "Days 29–35",
        phase: "Phase 5 — Zero-Mistake Planting Day & Root Collar Elevation",
        goal: "Plant the sapling with gentle root handling, perfect root collar elevation, and firm soil backfill.",
        notes: "Trusted Science (International Society of Arboriculture & ICAR): Planting too deep is the #1 killer of young trees worldwide! The root flare (where roots flare out from the trunk) must sit 1–2 inches ABOVE the soil surface. Burying trunk bark causes collar rot and trunk suffocation.",
        watchFor: [
            "Remove the plastic completely! Never plant with the polybag attached or left underneath.",
            "Loosen circling roots: If roots coil tightly like a bird's nest, gently tease the outer 1 cm apart so roots grow outward into garden soil.",
            "Best planting time: Plant in the cool late afternoon (4–6 PM) to give roots 12 hours of cool overnight recovery."
        ],
        items: [
            { day: "Day 29", observation: "Plant Day! Dig a small pocket in the center of the refilled pit matching the height of the nursery bag.", task: "Scoop out just enough enriched soil to hold the root ball." },
            { day: "Day 30", observation: "Slice the black nursery polybag down both sides with a sharp blade.", task: "Gently peel it away from the soil ball; dispose of plastic safely." },
            { day: "Day 31", observation: "Inspect the root ball: gently untangle any winding bottom roots with your fingertips.", task: "Fanning outer root tips outward prevents them from choking each other in later years." },
            { day: "Day 32", observation: "Lower the sapling into the center hole.", task: "Confirm that the root collar sits 2 inches (5 cm) above the surrounding ground level." },
            { day: "Day 33", observation: "Backfill enriched soil around the root ball in layers.", task: "Firm the soil gently with your shoes or hands to eliminate underground air pockets." },
            { day: "Day 34", observation: "Step back and check straightness from two angles.", task: "Straighten the trunk while the backfill is still workable." },
            { day: "Day 35", observation: "Confirm the trunk base is clean and no loose soil is piled against the green bark.", task: "The tree stands proudly and cleanly at its natural root flare height." }
        ]
    },
    {
        week: 6,
        days: "Days 36–42",
        phase: "Phase 6 — Saucer Water Basin & First 20-Liter Deep Drench",
        goal: "Construct a donut-shaped circular water basin and give the root zone its life-giving initial drench.",
        notes: "Trusted Science (ICAR-CAZRI Arid Agroforestry): A circular earthen ridge (bund) creates an irrigation saucer that traps water directly over the root ball. Deep initial watering (15–20 liters) forces out hidden air pockets and establishes hydraulic capillary contact between nursery soil and native ground.",
        watchFor: [
            "Saucer shape: Build the raised bund 10 cm high and 1 meter in diameter around the tree.",
            "Sinking soil check: If soil sinks or hollows appear near the stem after watering, immediately add dry soil to level it.",
            "Water the ring, not the trunk: Pour water gently into the basin away from direct contact with the trunk."
        ],
        items: [
            { day: "Day 36", observation: "Build a 10 cm (4-inch) high circular earth ridge around the tree, 1 meter in diameter.", task: "Shape the soil into a neat donut-like water basin." },
            { day: "Day 37", observation: "Give the first deep soak: pour 15 to 20 liters (approx. 2 full buckets) of clean water.", task: "Pour water slowly around the circular basin floor, letting it soak downward." },
            { day: "Day 38", observation: "Observe water absorption: water should soak downward smoothly within 10 to 15 minutes.", task: "Confirms proper soil porosity and elimination of air pockets." },
            { day: "Day 39", observation: "Inspect soil around the trunk for settling or crevices.", task: "Top up with 1 trowel of dry topsoil if minor depressions have formed." },
            { day: "Day 40", observation: "Check sapling posture: ensure the stem remains upright and hasn't leaned.", task: "Adjust gently if needed while soil is still soft." },
            { day: "Day 41", observation: "Monitor leaf turgor: fresh upright leaves indicate healthy water uptake.", task: "Leaves are hydrated and vibrant." },
            { day: "Day 42", observation: "Give second deep watering (10–15 liters) in the morning if top 2 inches feel dry.", task: "Keeps the baby root ball steadily hydrated during its first week in the ground." }
        ]
    },
    {
        week: 7,
        days: "Days 43–49",
        phase: "Phase 7 — The 3-3-3 Mulch Blanket & Bamboo Staking",
        goal: "Apply an organic protective blanket of mulch and install loose bamboo staking for wind stability.",
        notes: "Trusted Science (Arbor Day Foundation & TNAU Forestry): Follow the proven '3-3-3 Mulch Rule': 3 inches deep, in a 3-foot wide ring, keeping 3 inches of clear space away from the trunk. Never build a 'mulch volcano' against the bark! Tie to a sturdy stake in a loose figure-8 shape.",
        watchFor: [
            "Keep trunk clear: Mulch piled against wet bark suffocates living cells and attracts wood-boring beetle larvae.",
            "Flexible tie: Use wide cloth or bicycle rubber strips; never use thin wire or fishing line that strangles growing stems.",
            "Stake outside root ball: Drive the stake into native soil 15–20 cm away from the stem to avoid impaling roots."
        ],
        items: [
            { day: "Day 43", observation: "Collect clean organic mulch: dry leaves, rice straw, sugarcane bagasse, or coconut husks.", task: "Ensure mulch is free from plastic pieces or weed seeds." },
            { day: "Day 44", observation: "Spread mulch evenly across the 1-meter basin floor to a uniform depth of 3 inches (7–8 cm).", task: "Creates an insulating blanket that blocks weeds and keeps soil cool." },
            { day: "Day 45", observation: "Pull mulch back 3 inches away from the trunk.", task: "Create a clear, bare doughnut collar so air circulates freely around the bark." },
            { day: "Day 46", observation: "Drive a 6-foot strong bamboo or wooden pole 40 cm deep into firm ground.", task: "Place the pole 20 cm outside the root ball on the prevailing windward side." },
            { day: "Day 47", observation: "Secure the trunk to the stake with a soft cloth strip tied in a figure-8 knot.", task: "Figure-8 loop cushions the stem and prevents bark from scraping the wood." },
            { day: "Day 48", observation: "Test stem flexibility: the sapling should gently sway 2–3 inches in the wind.", task: "Gentle movement triggers hormonal trunk thickening." },
            { day: "Day 49", observation: "Check mulch moisture: reach under the mulch; the soil underneath feels cool and damp.", task: "Proves that evaporation loss is cut by up to 60%." }
        ]
    },
    {
        week: 8,
        days: "Days 50–56",
        phase: "Phase 8 — Tree Guard / Animal Shield Installation",
        goal: "Protect tender green bark and leaves from grazing goats, cattle, and rodents.",
        notes: "Trusted Science (ICAR-CAZRI Community Forestry): Grazing goats and stray cattle can strip young tree bark in minutes. A simple cylindrical wire mesh guard (1.5 meters tall) ensures survival rates exceed 95% in rural and semi-urban settings.",
        watchFor: [
            "Guard height: Tree guard should stand at least 4 to 5 feet high so livestock cannot reach over.",
            "Mesh spacing: Ensure wire mesh is narrow enough that goats cannot squeeze their snouts through.",
            "Anchor firmly: Tie or peg the guard securely so high winds don't knock it against the sapling."
        ],
        items: [
            { day: "Day 50", observation: "Source a circular wire mesh guard or assemble 4 bamboo slats wrapped with chicken wire mesh.", task: "Select a guard 1.5 meters tall and 60 cm in diameter." },
            { day: "Day 51", observation: "Position the tree guard centrally over the sapling.", task: "Ensure no wires scratch or touch green foliage." },
            { day: "Day 52", observation: "Drive 2 wooden anchoring stakes on opposite sides.", task: "Wire the guard firmly to the stakes so it cannot be tipped over." },
            { day: "Day 53", observation: "Inspect the interior: clear any rogue weeds that may sprout inside.", task: "Keep the inner basin spotless and weed-free." },
            { day: "Day 54", observation: "Check ventilation: open wire design allows maximum sunlight and breezy airflow.", task: "Ensures healthy transpiration while providing ironclad defense." },
            { day: "Day 55", observation: "Apply diluted neem oil spray (3 ml/L) on lower leaves if grasshoppers are spotted.", task: "Natural bitter spray deters chewing bugs." },
            { day: "Day 56", observation: "Two-month milestone review: verify the tree is well-anchored, guarded, and comfortably mulched!", task: "Celebrate successful initial establishment!" }
        ]
    },
    {
        week: 9,
        days: "Days 57–63",
        phase: "Phase 9 — Deep Root Hydration Routine",
        goal: "Establish a weekly deep-watering rhythm that trains taproots to dive deep underground rather than stay shallow.",
        notes: "Trusted Science (FAO Arid Zone Afforestation): Shallow daily watering produces weak, surface-level roots prone to drought kill. Deep watering 1 to 2 times a week forces roots to drill deep into subterranean moisture reserves.",
        watchFor: [
            "Deep soak rule: Apply 15–20 liters slowly once every 4 to 5 days during dry weather.",
            "Check soil with a finger: Poke 2 inches down under mulch; if cool and damp, do NOT water today!",
            "Yellowing leaves: If all leaves turn pale yellow and drop limp, reduce watering frequency immediately."
        ],
        items: [
            { day: "Day 57", observation: "Water basin deeply with 15 liters of water in early morning (6–8 AM).", task: "Early morning watering minimizes evaporation and hydrates roots before midday heat." },
            { day: "Day 58", observation: "Water percolates down into the deep subsoil, cooling the root zone.", task: "Moisture moves deep underground." },
            { day: "Day 59", observation: "Soil surface under mulch remains moist; no watering needed today.", task: "Let the soil breathe." },
            { day: "Day 60", observation: "Check root zone moisture with a wooden stick or finger 5 cm down; still happily damp.", task: "Resist the urge to overwater." },
            { day: "Day 61", observation: "Gentle soil drying encourages taproots to stretch downward in search of moisture.", task: "Natural root elongation in progress." },
            { day: "Day 62", observation: "Morning check: topsoil is turning light brown. Prepare for the next scheduled deep drench.", task: "Check leaves for crisp, healthy firmness." },
            { day: "Day 63", observation: "Water deeply with 15–20 liters.", task: "Maintain this steady 5-to-7 day watering cycle." }
        ]
    },
    {
        week: 10,
        days: "Days 64–70",
        phase: "Phase 10 — Scouting for Pests, Leaf Miners & Termite Prevention",
        goal: "Inspect tender new leaves for leaf miners, caterpillars, and ensure termite protection remains 100% active.",
        notes: "Trusted Science (TNAU Horticultural Pest Management): Young flush leaves are magnets for citrus leaf miners, aphids, and loopers. Spraying organic cold-pressed neem oil (5 ml per liter + 1 ml soap) creates an antifeedant barrier that repels pests safely.",
        watchFor: [
            "Silvery winding trails: Citrus leaf miners make shiny squiggly trails on fresh leaves; spray neem oil early morning.",
            "Mud tubes on bark: If subterranean termites build red mud tunnels on the lower trunk, brush them off and drench the basin with neem cake tea.",
            "Caterpillars: Pick off large leaf-eating caterpillars by hand wearing garden gloves."
        ],
        items: [
            { day: "Day 64", observation: "Inspect new leaf flushes on upper branch tips for winding silvery trails or aphids.", task: "Examine leaf undersides with a hand lens or clean eyes." },
            { day: "Day 65", observation: "Inspect the trunk base: confirm no termite mud tunnels are climbing up the wood.", task: "Ensure the root flare area is completely clean." },
            { day: "Day 66", observation: "Mix 5 ml organic neem oil + 1 ml mild dish soap in 1 liter of lukewarm water.", task: "Shake vigorously to create an emulsion." },
            { day: "Day 67", observation: "Spray neem solution thoroughly over both upper and lower leaf surfaces in the early morning.", task: "Coat leaves evenly; avoid spraying in hot midday sun." },
            { day: "Day 68", observation: "Check results: pests are repelled and beneficial ladybugs or spiders remain unharmed.", task: "Non-toxic organic defense verified." },
            { day: "Day 69", observation: "Pull any stray weeds that sprouted around the outer rim of the basin bund.", task: "Keep soil nutrients dedicated 100% to the tree." },
            { day: "Day 70", observation: "Refresh the 3-3-3 mulch blanket if birds or wind scattered dry leaves away.", task: "Maintain a full 3-inch protective blanket." }
        ]
    },
    {
        week: 11,
        days: "Days 71–77",
        phase: "Phase 11 — Pruning Below-Graft Rootstock Suckers",
        goal: "Remove wild rootstock suckers growing below the graft union on fruit trees to protect tree vigor.",
        notes: "Trusted Science (ICAR-IIHR Fruit Science): Grafted fruit trees (mango, guava, sapota, citrus) have a hardy rootstock base and a scion fruiting top. Any green shoot sprouting BELOW the graft union is a wild vampire shoot that will steal food and kill the desired fruiting tree if not removed!",
        watchFor: [
            "Locate the graft scar: Look for the diagonal healing scar 6 to 10 inches above soil level.",
            "Snip flush: Use clean, sharp pruning shears to cut off below-graft suckers flush against the trunk.",
            "Never touch shoots ABOVE the graft: These upper branches will form your future fruit canopy!"
        ],
        items: [
            { day: "Day 71", observation: "Locate the graft union scar on the lower trunk.", task: "Identify the boundary between rootstock and scion." },
            { day: "Day 72", observation: "Check for any small green shoots or leafy buds emerging below this scar line.", task: "Mark any wild shoots for removal." },
            { day: "Day 73", observation: "Disinfect pruning shears with rubbing alcohol or boiling water.", task: "Prevents introducing bacteria into fresh cut wounds." },
            { day: "Day 74", observation: "Snip off all suckers emerging from the rootstock base cleanly and flush against the bark.", task: "Cut cleanly without tearing the outer bark." },
            { day: "Day 75", observation: "Dab cut wounds with a tiny pinch of neem paste or turmeric powder.", task: "Natural antiseptic shields the cut surface." },
            { day: "Day 76", observation: "Inspect the upper scion branches: observe fresh, vigorous growth tips.", task: "Energy is now focused into the productive canopy." },
            { day: "Day 77", observation: "Record progress: your tree’s full nutrition is channeled exclusively into the prime fruiting canopy!", task: "Tree structure looks clean and well-balanced." }
        ]
    },
    {
        week: 12,
        days: "Days 78–84",
        phase: "Phase 12 — Month 3 Feeding (Vermicompost & Wood Ash Boost)",
        goal: "Feed the expanding root ball with balanced organic nutrients to fuel rapid vegetative canopy growth.",
        notes: "Trusted Science (TNAU Organic Farming Guidelines): At 90 days, the initial pit nutrients have been absorbed. Top-dressing with 3 to 5 kg of rich vermicompost and 100g wood ash (organic potassium) strengthens cell walls and expands canopy width.",
        watchFor: [
            "Scratch into topsoil: Lightly rake compost into the top 2 cm of soil under the mulch; avoid digging deep near roots.",
            "Apply along the drip line: Place nutrients under the outer edge of the leaf canopy, where feeder root tips actively drink.",
            "Water deeply immediately: Always drench the basin with water right after applying organic manure."
        ],
        items: [
            { day: "Day 78", observation: "Gently rake back the dry mulch blanket towards the outer perimeter of the basin.", task: "Expose the bare soil underneath for top-dressing." },
            { day: "Day 79", observation: "Measure 3 to 5 kg of dark, nutrient-rich vermicompost or composted cow manure.", task: "Ensure compost is well-aerated and crumbly." },
            { day: "Day 80", observation: "Sprinkle 100 grams of wood ash.", task: "Supplies organic potassium for thick cell walls and drought tolerance." },
            { day: "Day 81", observation: "Scatter the compost in a ring around the drip line (outer edge of branch tips).", task: "Directs food to where new feeder roots are actively exploring." },
            { day: "Day 82", observation: "Scratch the compost gently into the top 2 cm of soil using a small hand cultivator.", task: "Incorporate lightly without bruising shallow root hairs." },
            { day: "Day 83", observation: "Rake the 3-3-3 mulch blanket back into place over the fertilized soil.", task: "Restores the moisture-retaining cover." },
            { day: "Day 84", observation: "Give a generous 20-liter deep soak to dissolve organic nutrients into the root zone.", task: "Feeds nutrients directly to thirsty feeder roots." }
        ]
    },
    {
        week: 13,
        days: "Days 85–91",
        phase: "Phase 13 — Basin Ring Widening & Root Expansion",
        goal: "Expand the circular water basin outward to match the growing tree canopy and encourage wider root spread.",
        notes: "Trusted Science (FAO Agroforestry Technical Papers): Tree roots grow outward at twice the width of the above-ground leaf canopy. Widening the saucer basin by 20–30 cm keeps water delivered directly to expanding feeder root tips.",
        watchFor: [
            "Widen the basin: Move the outer earth bund outward so it sits slightly beyond the current leaf canopy width.",
            "Do not disturb internal roots: Never dig deeply inside the original root ball zone.",
            "Level the basin floor: Ensure water spreads evenly across the entire widened ring."
        ],
        items: [
            { day: "Day 85", observation: "Measure the current width of the tree’s leaf canopy.", task: "Notice how much new branching has occurred over the last 3 months." },
            { day: "Day 86", observation: "Rebuild the circular bund 20 cm wider than the current canopy drip line.", task: "Create an expanded irrigation ring." },
            { day: "Day 87", observation: "Flatten and clear the expanded basin floor of competing weeds and grass.", task: "Remove any aggressive lawn grass creeping into the basin." },
            { day: "Day 88", observation: "Add a thin layer of dry straw mulch over the freshly exposed basin area.", task: "Extends the mulch shield across the larger root zone." },
            { day: "Day 89", observation: "Test water distribution: fill basin with 20 liters; ensure water reaches outer edges.", task: "Confirm even water leveling across the wider basin." },
            { day: "Day 90", observation: "Inspect trunk girth: the main trunk is visibly thicker and sturdier than on planting day!", task: "Wood density is increasing nicely." },
            { day: "Day 91", observation: "Loosen the bamboo stake tie slightly if the trunk has thickened.", task: "Prevents cord constriction as stem diameter grows." }
        ]
    },
    {
        week: 14,
        days: "Days 92–98",
        phase: "Phase 14 — Summer Defense: Trunk Whitewashing & Sun Shield",
        goal: "Protect tender trunk bark from intense sunscald and cracking during hot, dry summer months.",
        notes: "Trusted Science (ICAR-CAZRI & University of California Horticulture): Painting young tree trunks with diluted white agricultural lime or water-based white latex paint reflects radiant solar heat, keeping trunk temperatures up to 8°C cooler and preventing bark splitting and borer attack.",
        watchFor: [
            "Safe white wash: Mix slaked lime (chuna) with water and a pinch of copper oxychloride or turmeric. Never use oil-based paints!",
            "Paint height: Paint from ground level up to 45 cm (1.5 feet) high on the main trunk.",
            "Mulch check: Ensure the mulch blanket is a full 3 inches deep before peak summer heat sets in."
        ],
        items: [
            { day: "Day 92", observation: "Mix 100g agricultural slaked lime (calcium hydroxide) in 1 liter of water.", task: "Creates a milky white reflective paint." },
            { day: "Day 93", observation: "Add 2 grams of turmeric powder or copper oxychloride.", task: "Acts as an organic antiseptic shield against bark borers." },
            { day: "Day 94", observation: "Using a soft 2-inch paintbrush, paint the lower 45 cm of the main trunk white.", task: "Coat the bark evenly on all four sides." },
            { day: "Day 95", observation: "Allow whitewash to dry into a bright chalky reflective coating.", task: "Bounces harsh UV rays away from delicate bark cambium cells." },
            { day: "Day 96", observation: "Verify moisture under mulch: replenish dry leaves or straw if thinned under 3 inches.", task: "Maintains optimal soil coolness during blistering afternoons." },
            { day: "Day 97", observation: "Increase summer watering: during extreme heat (>35°C), deep water every 3–4 days.", task: "Water early in the morning before heat rises." },
            { day: "Day 98", observation: "Observe leaf vitality: whitewashed trees show zero sunburn and zero midday wilting stress.", task: "Sapling is safely shielded against summer heatwaves." }
        ]
    },
    {
        week: 15,
        days: "Days 99–105",
        phase: "Phase 15 — Pre-Monsoon Manure Dressing & Drainage Inspection",
        goal: "Prepare the tree for heavy monsoon rains by opening basin drainage and providing pre-rain organic nutrition.",
        notes: "Trusted Science (TNAU Agritech Portal): Heavy rains can waterlog the saucer basin, causing root asphyxiation if water stands for days. Cutting a small notch in the outer bund allows excess monsoon deluge to drain safely while feeding roots.",
        watchFor: [
            "Prevent standing water: During monsoon, create an outlet spillway in the circular bund so excess rainwater flows away.",
            "Pre-monsoon compost: Apply 5 kg compost now so early monsoon rains carry rich organic nutrients down to hungry roots.",
            "Fungal shield: Spray neem oil or 0.1% copper fungicide before continuous overcast rain begins."
        ],
        items: [
            { day: "Day 99", observation: "Cut a 10 cm opening in the downslope side of the circular bund.", task: "Acts as an emergency rain spillway to prevent standing puddles." },
            { day: "Day 100", observation: "Scatter 5 kg decomposed cow manure or compost evenly around the drip line.", task: "Top-dress the root perimeter before seasonal rains arrive." },
            { day: "Day 101", observation: "Lightly incorporate the manure into the topsoil so it doesn't wash away in heavy downpours.", task: "Rake smoothly into the top layer." },
            { day: "Day 102", observation: "Inspect bamboo stake: reinforce the stake in the ground.", task: "Ensures storm winds cannot uproot the stake." },
            { day: "Day 103", observation: "Adjust figure-8 ties: ensure the trunk has enough room to sway in high gusty winds.", task: "Prevents ties from breaking under wind strain." },
            { day: "Day 104", observation: "Clear fallen wet debris around the trunk.", task: "Stops mold and fungal spores from colonizing wet bark." },
            { day: "Day 105", observation: "Watch monsoon rains revitalize the young tree with nitrogen-rich rainwater!", task: "Vigorous fresh green shoot flushes appear across the canopy." }
        ]
    },
    {
        week: 16,
        days: "Days 106–112",
        phase: "Phase 16 — 1-Year Milestone Review & Natural Canopy Shaping",
        goal: "Celebrate the 1-year establishment milestone, remove temporary stakes, and guide natural canopy architecture.",
        notes: "Trusted Science (Forest Research Institute & ICAR): At 1 year, a well-cared-for sapling has anchored deep permanent taproots and can stand strong without artificial stakes. Light corrective pruning of crossing or broken branches builds a strong, open, storm-resistant tree framework for decades of life.",
        watchFor: [
            "Remove the nursery stake: If the trunk is sturdy and self-supporting, remove the bamboo stake to let the tree build full wood density.",
            "3-to-4 scaffold branches: Select 3 to 4 well-spaced primary branches growing in different compass directions.",
            "Celebrate: Your tree is officially permanently established, cleaning air, cooling the soil, and preparing for future blooms!"
        ],
        items: [
            { day: "Day 106", observation: "Inspect trunk firmness: untie the bamboo stake and test stem resistance with your hand.", task: "Check if the tree stands rigid and self-supporting." },
            { day: "Day 107", observation: "If the tree stands firm on its own roots, carefully pull out the temporary bamboo stake.", task: "Remove the stake so the tree builds natural wood toughness." },
            { day: "Day 108", observation: "Inspect canopy structure: identify 3 to 4 strong, balanced scaffold branches.", task: "Select branches with wide, strong 45-degree crotch angles." },
            { day: "Day 109", observation: "Snip any dead, damaged, or inward-crossing branches with clean pruning shears.", task: "Open the center to allow bright sunlight and breeze into the crown." },
            { day: "Day 110", observation: "Refresh the permanent organic mulch ring around the tree’s expanded 1.5-meter drip zone.", task: "Maintain soil health for the coming years." },
            { day: "Day 111", observation: "Record Year 1 tree height, trunk girth, and health notes in your LeafLine gardening log!", task: "Celebrate this rewarding milestone." },
            { day: "Day 112", observation: "Celebrate your success! Your tree is now self-sufficient, climate-resilient, and thriving.", task: "Enjoy shade, fresh oxygen, and future fruits for generations to come!" }
        ]
    }
];

fs.writeFileSync('scratch/tree_schedule.json', JSON.stringify(TREE_PLANTING_SCHEDULE, null, 2));
console.log('Tree planting schedule written successfully. Weeks:', TREE_PLANTING_SCHEDULE.length);
