const fs = require('fs');

let html = fs.readFileSync('frontend/RoadMap.html', 'utf8');

// 1. Read TREE_PLANTING_SCHEDULE json
const treeScheduleJson = fs.readFileSync('scratch/tree_schedule.json', 'utf8');
const treeScheduleCode = `\n        // ── Tree Planting 16-Week Day-by-Day Master Care Schedule ───────────\n        const TREE_PLANTING_SCHEDULE = ${treeScheduleJson};\n`;

// Insert TREE_PLANTING_SCHEDULE right before CROP_WEEKLY_SCHEDULES
if (!html.includes('const TREE_PLANTING_SCHEDULE =')) {
    html = html.replace('        // Map of available crop weekly schedules', treeScheduleCode + '\n        // Map of available crop weekly schedules');
}

// 2. Update CROP_WEEKLY_SCHEDULES
html = html.replace(
    `        const CROP_WEEKLY_SCHEDULES = {
            tomato: TOMATO_WEEKLY_SCHEDULE,
            chilli: CHILLI_WEEKLY_SCHEDULE,
            brinjal: BRINJAL_WEEKLY_SCHEDULE,
            beans: BEANS_WEEKLY_SCHEDULE,
            cucumber: CUCUMBER_WEEKLY_SCHEDULE,
        };`,
    `        const CROP_WEEKLY_SCHEDULES = {
            tomato: TOMATO_WEEKLY_SCHEDULE,
            chilli: CHILLI_WEEKLY_SCHEDULE,
            brinjal: BRINJAL_WEEKLY_SCHEDULE,
            beans: BEANS_WEEKLY_SCHEDULE,
            cucumber: CUCUMBER_WEEKLY_SCHEDULE,
            tree_planting: TREE_PLANTING_SCHEDULE,
        };`
);

// 3. Update selectedCropWeek
html = html.replace(
    `        const selectedCropWeek = {
            tomato: 1,
            chilli: 1,
            brinjal: 1,
            beans: 1,
            cucumber: 1,
        };`,
    `        const selectedCropWeek = {
            tomato: 1,
            chilli: 1,
            brinjal: 1,
            beans: 1,
            cucumber: 1,
            tree_planting: 1,
        };`
);

// 4. Update cropViewMode
html = html.replace(
    `        const cropViewMode = {
            tomato: 'weekly',
            chilli: 'weekly',
            brinjal: 'weekly',
            beans: 'weekly',
            cucumber: 'weekly',
        };`,
    `        const cropViewMode = {
            tomato: 'weekly',
            chilli: 'weekly',
            brinjal: 'weekly',
            beans: 'weekly',
            cucumber: 'weekly',
            tree_planting: 'weekly',
        };`
);

// 5. Update CROP_ESSENTIALS with tree_planting
const treeEssentialsCode = `            tree_planting: {
                title: "Tree Planting Essentials & Trusted Care Rules",
                source: "ICAR-CAZRI, Forest Research Institute (FRI) & FAO Guidelines",
                cards: [
                    { icon: "fa-ruler-combined text-emerald-500", label: "Dig Wide, Not Deep", text: "Dig pit 2–3 times wider than the root ball, but only as deep as the root ball. Never bury the root collar!" },
                    { icon: "fa-sun text-amber-500", label: "Sun Solarization (10 Days)", text: "Leave the dug pit open to hot sun for 10–15 days to naturally kill termites, nematodes, and soil fungi." },
                    { icon: "fa-shield-halved text-rose-500", label: "Neem Cake & Bio-Shield", text: "Mix topsoil with 15kg compost + 500g neem cake (stops termites) + 50g Trichoderma bio-fungicide." },
                    { icon: "fa-circle-dot text-blue-500", label: "Saucer Water Basin", text: "Build a 1-meter circular soil ring around the tree to catch 15–20 liters of deep soaking water." },
                    { icon: "fa-layer-group text-amber-600", label: "The 3-3-3 Mulch Rule", text: "3 inches deep, 3-foot wide ring, 3 inches away from trunk bark. Never pile mulch against the trunk!" },
                    { icon: "fa-lines-leaning text-purple-500", label: "Flexible Staking", text: "Tie trunk loosely to stake in a figure-8 with soft rubber/cloth. Allow slight natural sway to build strong wood." }
                ]
            },
            cucumber: {`;

html = html.replace('            cucumber: {', treeEssentialsCode);

// 6. Update CROP_ICONS
html = html.replace(
    `        const CROP_ICONS = {
            tomato: '🍅',
            chilli: '🌶️',
            brinjal: '🍆',
            beans: '🫘',
            cucumber: '🥒'
        };`,
    `        const CROP_ICONS = {
            tomato: '🍅',
            chilli: '🌶️',
            brinjal: '🍆',
            beans: '🫘',
            cucumber: '🥒',
            tree_planting: '🌳'
        };`
);

// 7. Update CROP_SNAPSHOTS
const treeSnapshotCode = `            cucumber: {
                emoji: '🥒',
                title: 'Cucumber Agronomy',
                botanical: 'Cucumis sativus L.',
                days: '60–70 Days',
                temp: '24°C – 30°C',
                spacing: '90 × 60 cm (trellis)',
                water: 'Consistent moisture essential for crisp, sweet fruit. Avoid wetting leaves.'
            },
            tree_planting: {
                emoji: '🌳',
                title: 'Tree Planting Guide',
                botanical: 'Universal Tree & Sapling Care',
                days: 'Year 1 Care',
                temp: 'All Seasons (Monsoon ideal)',
                spacing: '3–6 Meters apart',
                water: 'Deep saucer basin soak (15–20L) 1–2 times weekly; never water trunk directly.'
            }`;
html = html.replace(/cucumber:\s*\{\s*emoji:\s*'🥒'[\s\S]*?water:\s*'[^']*'\s*\}/, treeSnapshotCode);

// 8. Update FALLBACK_CROPS
const treeFallbackCrop = `            { crop_id: 'cucumber', common_names: ['Cucumber'], scientific_name: 'Cucumis sativus L.', stage_count: 4, evidence_status: 'verified' },
            { crop_id: 'tree_planting', common_names: ['Tree Planting Guide', 'Trees & Saplings'], scientific_name: 'Universal Sapling & Fruit Tree Care', stage_count: 8, evidence_status: 'verified' }`;
html = html.replace("            { crop_id: 'cucumber', common_names: ['Cucumber'], scientific_name: 'Cucumis sativus L.', stage_count: 4, evidence_status: 'verified' }", treeFallbackCrop);

// 9. Update getFallbackRoadmap with tree_planting
const treeFallbackRoadmap = `                brinjal: {
                    crop_id: 'brinjal',
                    common_names: ['Brinjal', 'Eggplant'],
                    scientific_name: 'Solanum melongena L.',
                    evidence_status: 'verified',
                    identity_note: 'Solanum melongena (Brinjal) — 16-Week Day-by-Day Lifecycle & Agronomic Care.',
                    supported_methods: ['transplanting', 'ipm_borer_management'],
                    applicability_note: 'TNAU Agritech and ICAR package of practices.',
                    focus_stages: ['nursery', 'transplanting', 'flowering', 'harvesting'],
                    stages: [
                        { stage_id: 'nursery', stage_name: 'Nursery Stage', timing_display: 'Days 1–30', timing_basis: 'days_after_sowing', evidence_status: 'verified', critical_tasks: ['Seed treatment', 'Raised bed care'], pest_disease_watch: ['Damping off'] },
                        { stage_id: 'transplanting', stage_name: 'Field Transplanting', timing_display: 'Days 30–45', timing_basis: 'days_after_transplanting', evidence_status: 'verified', critical_tasks: ['Spacing 60x60cm', 'Basal manure application'], pest_disease_watch: ['Bacterial Wilt'] },
                        { stage_id: 'flowering', stage_name: 'Branching & Fruit Setting', timing_display: 'Days 45–90', timing_basis: 'days_after_transplanting', evidence_status: 'verified', critical_tasks: ['Install pheromone traps for shoot & fruit borer (10/ha)', 'Clip wilted shoots regularly'], pest_disease_watch: ['Shoot & Fruit Borer (Leucinodes orbonalis)', 'Little leaf'] },
                        { stage_id: 'harvesting', stage_name: 'Harvesting', timing_display: 'Days 90–120', timing_basis: 'days_after_transplanting', evidence_status: 'verified', critical_tasks: ['Harvest tender glossy fruits', 'Pick every 4–5 days'], pest_disease_watch: ['Phomopsis blight'] }
                    ]
                },
                tree_planting: {
                    crop_id: 'tree_planting',
                    common_names: ['Tree Planting Guide', 'Trees & Saplings'],
                    scientific_name: 'Universal Sapling & Fruit Tree Care',
                    evidence_status: 'verified',
                    identity_note: 'Universal 8-Stage Tree & Sapling Planting Master Guide — Grounded in ICAR-CAZRI, Forest Research Institute (FRI) & FAO Guidelines.',
                    supported_methods: ['pit_solarization', 'organic_neem_shield', 'saucer_basin_irrigation', '3-3-3_mulching', 'flexible_staking'],
                    applicability_note: 'Grounded in ICAR-CAZRI and FAO agroforestry manuals; applicable to fruit trees (mango, guava, citrus, sapota, pomegranate) and shade/forest trees across all regions.',
                    focus_stages: ['siting', 'pit_prep', 'soil_enrichment', 'planting', 'irrigation', 'mulching', 'staking', 'year1_care'],
                    stages: [
                        { stage_id: 'siting', stage_name: 'Site Selection, Sunlight & Distance Planning', timing_display: '3–4 Weeks Before Planting', timing_basis: 'planning_phase', evidence_status: 'verified', critical_tasks: ['Pick a planting spot with 6–8 hours direct sunshine', 'Keep 15–20 feet away from house walls, septic tanks & pipes', 'Test soil drainage in a 1-foot test hole (should drain in 2–3 hours)'], pest_disease_watch: ['Waterlogged standing water', 'Overhead electrical power lines'] },
                        { stage_id: 'pit_prep', stage_name: 'Digging the 3×3×3 Pit & Sun Solarization', timing_display: '2–3 Weeks Before Planting', timing_basis: 'pit_preparation', evidence_status: 'verified', critical_tasks: ['Dig square pit 1m × 1m × 1m (3ft × 3ft × 3ft)', 'Keep dark topsoil in one pile, subsoil in another', 'Leave open to hot sun for 10–15 days to sterilize soil naturally', 'Break bottom hardpan with crowbar so taproots can dive deep'], pest_disease_watch: ['Subterranean termite nests in soil'] },
                        { stage_id: 'soil_enrichment', stage_name: 'Pit Soil Mixture & Organic Termite Shield', timing_display: '3–5 Days Before Planting', timing_basis: 'soil_enrichment', evidence_status: 'verified', critical_tasks: ['Mix topsoil with 15–20 kg aged compost or vermicompost', 'Add 500g Neem Cake powder (repels termites & root grubs)', 'Add 50g Trichoderma bio-fungicide (stops root rot)', 'Add 100g bone meal or rock phosphate for rapid root branching', 'Refill pit and water lightly to settle loose earth before planting'], pest_disease_watch: ['Termites', 'White root grubs'] },
                        { stage_id: 'planting', stage_name: 'Zero-Mistake Sapling Planting Day', timing_display: 'Day 0 (Planting Day - 4–6 PM)', timing_basis: 'planting_day', evidence_status: 'verified', critical_tasks: ['Carefully slice and remove nursery polybag without breaking root ball', 'Untangle winding bottom roots so they fan outward', 'CRUCIAL: Position root collar 2 inches ABOVE soil line (never bury trunk!)', 'Backfill enriched soil and press down gently to remove air pockets', 'Plant in late afternoon (4–6 PM) for cool overnight root recovery'], pest_disease_watch: ['Collar rot (bark rotting from being buried too deep — #1 killer!)'] },
                        { stage_id: 'irrigation', stage_name: 'The Saucer Basin & First 20-Liter Deep Drench', timing_display: 'Immediately on Planting Day', timing_basis: 'post_planting', evidence_status: 'verified', critical_tasks: ['Build a 10 cm high circular earth ring (bund) 1 meter wide around sapling', 'Slowly pour 15–20 liters (approx. 2 buckets) into the saucer basin', 'Water inside the ring; never splash water directly against trunk bark', 'If soil sinks after watering, top up with 1 trowel of dry soil immediately'], pest_disease_watch: ['Sinkholes causing tree to tilt'] },
                        { stage_id: 'mulching', stage_name: 'The 3-3-3 Mulching Blanket (Natural Soil Shield)', timing_display: 'Day 1 After Planting', timing_basis: 'mulching_phase', evidence_status: 'verified', critical_tasks: ['Spread dry leaves, straw or wood chips 3 inches deep in a 3-foot ring', 'Keep 3 inches of CLEAR SPACE away from the trunk bark (no mulch volcano!)', 'Mulch cuts water evaporation by 60%, blocks weeds, and keeps roots cool'], pest_disease_watch: ['Mulch volcano suffocating trunk bark'] },
                        { stage_id: 'staking', stage_name: 'Bamboo Staking & Soft Figure-8 Wind Ties', timing_display: 'Days 1–7 After Planting', timing_basis: 'staking_phase', evidence_status: 'verified', critical_tasks: ['Drive 6-foot bamboo stake 40 cm deep, 20 cm outside the root ball', 'Tie trunk to stake with soft cloth or bicycle rubber strip in a figure-8 knot', 'Allow 2–3 inches of natural sway in wind to stimulate thick trunk growth', 'NEVER use thin wire or tight cord that strangles growing bark'], pest_disease_watch: ['Bark abrasion from tight ties'] },
                        { stage_id: 'year1_care', stage_name: 'Year 1 Seasonal Care, Tree Guard & Protection', timing_display: 'Months 1–12 (Full First Year)', timing_basis: 'first_year_management', evidence_status: 'verified', critical_tasks: ['Install 4-to-5 foot wire mesh tree guard to stop grazing goats & cattle', 'Deep water 15–20 liters 1–2 times weekly during dry weather (no shallow sprinkles)', 'Fruit trees: Snip off any wild suckers growing below the graft union immediately', 'Summer defense: Paint lower trunk white with slaked lime (chuna) to reflect heat', 'Top-dress with 5 kg vermicompost every 3 months around the drip line'], pest_disease_watch: ['Grazing animals', 'Wild rootstock suckers', 'Sunburn trunk cracking'] }
                    ]
                }`;
html = html.replace(/brinjal:\s*\{\s*crop_id:\s*'brinjal'[\s\S]*?pest_disease_watch:\s*\['Phomopsis blight'\]\s*\}\s*\]\s*\}/, treeFallbackRoadmap);

// 10. Update subP and duration texts
html = html.replace(
    "setText(subP, 'Viewing ' + weekData.days + ' of the ' + (cropId === 'chilli' ? '120 Day Chilli' : (cropId === 'brinjal' ? '120 Day Brinjal' : (cropId === 'beans' ? '120 Day Beans' : (cropId === 'cucumber' ? '120 Day Cucumber' : '110+ Day Tomato')))) + ' Lifecycle');",
    "setText(subP, 'Viewing ' + weekData.days + ' of the ' + (cropId === 'tree_planting' ? 'Year 1 Tree Planting & Sapling Care Journey' : (cropId === 'chilli' ? '120 Day Chilli Lifecycle' : (cropId === 'brinjal' ? '120 Day Brinjal Lifecycle' : (cropId === 'beans' ? '120 Day Beans Lifecycle' : (cropId === 'cucumber' ? '120 Day Cucumber Lifecycle' : '110+ Day Tomato Lifecycle'))))));"
);

html = html.replace(
    "const totalDays = (cropId === 'chilli' || cropId === 'brinjal' || cropId === 'beans' || cropId === 'cucumber') ? '120' : '110+';",
    "const totalDays = (cropId === 'tree_planting') ? 'Year 1' : ((cropId === 'chilli' || cropId === 'brinjal' || cropId === 'beans' || cropId === 'cucumber') ? '120' : '110+');"
);

html = html.replace(
    "setText(wTxt, '📅 1-Week at a Time Schedule (Days 1–' + totalDays + ')');",
    "setText(wTxt, (cropId === 'tree_planting' ? '📅 1-Week at a Time Schedule (Year 1 Care Journey)' : '📅 1-Week at a Time Schedule (Days 1–' + totalDays + ')'));"
);

html = html.replace(
    "setText(countBadge, totalWeeks + ' Weeks · ' + ((cropId === 'chilli' || cropId === 'brinjal' || cropId === 'beans' || cropId === 'cucumber') ? '120 Days' : '110+ Days') + ' · ' + (cropId === 'cucumber' ? '12 Phases' : (cropId === 'beans' ? '10 Phases' : '8 Phases')));",
    "setText(countBadge, totalWeeks + ' Weeks · ' + (cropId === 'tree_planting' ? '12-Month Care · 8 Scientific Phases' : (((cropId === 'chilli' || cropId === 'brinjal' || cropId === 'beans' || cropId === 'cucumber') ? '120 Days' : '110+ Days') + ' · ' + (cropId === 'cucumber' ? '12 Phases' : (cropId === 'beans' ? '10 Phases' : '8 Phases')))));"
);

// 11. Update grid column count in loadCrops
html = html.replace(
    "grid.className = 'grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3.5';",
    "grid.className = 'grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5';"
);

// 12. Add Tree Planting callout in left sidebar right under Navigation Card
const sidebarTreeCard = `            <!-- Tree Planting Quick Action Card -->
            <div class="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-3xl p-4 border border-emerald-200/80 shadow-2xs space-y-2.5">
                <div class="flex items-center justify-between">
                    <span class="text-xs font-bold text-emerald-900 flex items-center gap-1.5">
                        <i class="fa-solid fa-tree text-emerald-600"></i>
                        <span>Tree Planting Guide</span>
                    </span>
                    <span class="text-[9px] font-extrabold uppercase px-1.5 py-0.5 rounded bg-emerald-200/60 text-emerald-800">ICAR & FAO</span>
                </div>
                <p class="text-[11px] text-emerald-800/80 leading-relaxed">
                    Planning to plant fruit or shade trees? Follow the 8-stage zero-mistake master guide.
                </p>
                <button type="button" onclick="selectCrop('tree_planting')" class="w-full py-2 px-3 rounded-xl bg-white hover:bg-emerald-100/80 border border-emerald-300/80 text-emerald-800 text-xs font-bold transition flex items-center justify-center gap-2 shadow-2xs">
                    <i class="fa-solid fa-seedling text-emerald-600"></i>
                    <span>Open Tree Planting Guide</span>
                </button>
            </div>\n\n`;

if (!html.includes('<!-- Tree Planting Quick Action Card -->')) {
    html = html.replace('            <!-- Active Crop Agronomy Snapshot Widget -->', sidebarTreeCard + '            <!-- Active Crop Agronomy Snapshot Widget -->');
}

// 13. Update crop selector heading subtitle
html = html.replace(
    'Explore weekly day-by-day actions grounded in ICAR & TNAU agricultural research.',
    'Explore weekly day-by-day actions for crops and trees grounded in ICAR, TNAU & FAO research.'
);

fs.writeFileSync('frontend/RoadMap.html', html, 'utf8');
console.log('frontend/RoadMap.html updated successfully.');
