"""Poultry Disease Knowledge Base & Clinical Reference.

Covers all 14 major poultry diseases:
- 5 Active Model Classes (trained & validated in MobileNetV2 with 91.32% held-out test accuracy)
- 9 Educational / Reference Classes (cataloged with full veterinary clinical profiles;
  identified as requiring additional labeled image datasets for model retraining)
"""

# Exact class order used during training (alphabetical folder order):
# 0: Coccidiosis, 1: Fowl Pox, 2: Healthy, 3: Newcastle Disease, 4: Salmonella
CLASS_NAMES = ["Coccidiosis", "Fowl Pox", "Healthy", "Newcastle Disease", "Salmonella"]

# Complete dictionary of all 14 poultry diseases
DISEASE_INFO = {
    # =========================================================================
    # 1. COCCIDIOSIS (Active Model Prediction)
    # =========================================================================
    "Coccidiosis": {
        "name": "Coccidiosis",
        "model_supported": True,
        "training_status": "Active Model Prediction (MobileNetV2 · 100.0% Precision, 94.64% Recall)",
        "severity": "Moderate to High",
        "badge": "warning",
        "icon": "🦠",
        "scientific_name": "Eimeria tenella / necatrix / acervulina",
        "pathogen_type": "Protozoan Endoparasite (Apicomplexa)",
        "incubation": "4 – 7 days",
        "mortality": "Moderate (up to 50% in untreated broiler flocks)",
        "transmission": "Fecal-oral route through ingestion of sporulated oocysts in damp litter, contaminated drinkers, feed, or soil.",
        "summary": (
            "Coccidiosis is a destructive intestinal disease caused by microscopic protozoa of the genus Eimeria. "
            "The parasites invade and rupture epithelial cells in the intestinal mucosa and caeca, causing severe haemorrhage, "
            "malabsorption of nutrients, electrolyte imbalance, and high susceptibility to secondary necrotic enteritis."
        ),
        "head_signs": "Pale, shrunken comb and wattles resulting from acute blood loss and anemia; dull, sunken eyes; drooping head.",
        "body_signs": "Severely ruffled feathers, hunched 'penguin-like' posture, drooping wings, extreme lethargy, bloody or dark reddish-brown diarrhea.",
        "fecal_signs": "Bloody, mucoid, reddish-brown or orange-tinted watery diarrhea containing sloughed mucosal tissue.",
        "symptoms": [
            "Bloody or deep brownish-red watery diarrhea",
            "Huddled posture with ruffled plumage and closed eyes",
            "Severe drop in feed consumption and water intake",
            "Pale comb and wattles indicative of severe anemia",
            "Rapid dehydration, body weight loss, and stunted flock growth",
        ],
        "prevention": [
            "Maintain dry, well-aerated litter (< 25% moisture); promptly replace damp bedding around drinker lines",
            "Sanitize bell drinkers and feed troughs daily with poultry-approved disinfectants",
            "Administer coccidiostats (e.g., ionophores or chemical anticoccidials) in feed according to veterinary rotation programs",
            "Vaccinate replacement pullets at hatchery with attenuated live oocyst vaccines",
            "Avoid flock overcrowding to keep oocyst concentration in litter below pathogenic thresholds",
        ],
        "recommendation": (
            "Immediately isolate birds exhibiting bloody diarrhea. Initiate anticoccidial water treatment "
            "(e.g., Amprolium, Toltrazuril, or Sulfadimethoxine) under the guidance of a licensed poultry veterinarian. "
            "Replace wet litter immediately to break the oocyst life cycle."
        ),
    },

    # =========================================================================
    # 2. HEALTHY CHICKEN (Active Model Prediction)
    # =========================================================================
    "Healthy": {
        "name": "Healthy Chicken / No Visible Disease",
        "model_supported": True,
        "training_status": "Active Model Prediction (MobileNetV2 · 84.85% Precision, 93.33% Recall)",
        "severity": "Normal / Healthy",
        "badge": "success",
        "icon": "✅",
        "scientific_name": "Normal Avian Microflora & Physiology",
        "pathogen_type": "None Detected (Physiologically Balanced)",
        "incubation": "N/A",
        "mortality": "0%",
        "transmission": "N/A — Bird exhibits physiological equilibrium.",
        "summary": (
            "The specimen displays normal physiological characteristics. Droppings exhibit standard digestive breakdown "
            "with firm consistency and characteristic white urate caps. Head, eyes, comb, plumage, and posture show no pathognomonic "
            "lesions, swelling, or clinical abnormalities."
        ),
        "head_signs": "Bright red, supple comb and wattles; clear, alert, circular eyes; clean nostrils without discharge; dry, smooth beak.",
        "body_signs": "Smooth, well-preened feathers; upright, active, alert posture; vigorous foraging; clean cloacal vent without pasty staining.",
        "fecal_signs": "Firm, well-formed brown or greyish droppings capped with crisp white uric acid crystals; no blood, foul odor, or excessive liquid.",
        "symptoms": [
            "Firm, well-formed droppings with crisp white uric acid caps",
            "Bright, responsive eyes with active foraging and normal social flock behavior",
            "Vibrant red comb and wattles free from scabs, swelling, or cyanosis",
            "Consistent daily feed and water consumption matching breed benchmarks",
            "Steady egg laying or expected broiler weight gain rates",
        ],
        "prevention": [
            "Maintain strict perimeter farm biosecurity: vehicle disinfection tire baths, boot dips, and restricted visitor access",
            "Provide balanced, mycotoxin-free mash or crumble feed matching the flock's specific production age",
            "Ensure continuous 24/7 access to fresh, cool, potable drinking water tested regularly for coliforms",
            "Adhere strictly to the recommended national poultry vaccination calendar (Marek's, ND, IBD, IB)",
            "Maintain proper shed ventilation to prevent ammonia buildup exceeding 20 ppm",
        ],
        "recommendation": (
            "Flock appears healthy. Maintain standard flock management, nutrition, and routine biosecurity protocols. "
            "Conduct weekly visual flock health walks and monitor feed/water intake logs."
        ),
    },

    # =========================================================================
    # 3. NEWCASTLE DISEASE (Active Model Prediction)
    # =========================================================================
    "Newcastle Disease": {
        "name": "Newcastle Disease (Ranikhet)",
        "model_supported": True,
        "training_status": "Active Model Prediction (MobileNetV2 · 82.93% Precision, 89.47% Recall)",
        "severity": "Critical / Emergency",
        "badge": "danger",
        "icon": "⚠️",
        "scientific_name": "Avian orthoavulavirus 1 (Paramyxovirus Type 1)",
        "pathogen_type": "Single-stranded RNA Paramyxovirus",
        "incubation": "2 – 15 days (average 4 – 6 days)",
        "mortality": "Extremely High (up to 90% – 100% in velogenic viscerotropic/neurotropic strains)",
        "transmission": "Inhalation of airborne viral aerosols, direct bird-to-bird contact, ingestion of infective droppings, contaminated equipment, footwear, feed, and wild migratory birds.",
        "summary": (
            "Newcastle Disease is a devastating, highly contagious, notifiable viral disease affecting domestic poultry and other avian species. "
            "It manifests in respiratory, nervous, and gastrointestinal systems simultaneously. Velogenic strains cause rapid flock-wide mortality, "
            "gasping, torticollis (twisted neck), facial edema, and emerald-green diarrhea."
        ),
        "head_signs": "Marked facial and peri-orbital swelling; cyanotic (dark purplish) comb and wattles; twisted neck (torticollis); conjunctivitis; nasal discharge and open-mouth gasping.",
        "body_signs": "Tremors, wing and leg paralysis, loss of equilibrium, circle-walking, complete prostration, profuse watery emerald-green diarrhea.",
        "fecal_signs": "Profuse, watery, bright emerald-green or yellowish-green diarrhea with high fluid volume.",
        "symptoms": [
            "Profuse emerald-green or bright greenish-yellow watery diarrhea",
            "Severe respiratory distress: open-mouth gasping, coughing, rattling rales, and sneezing",
            "Neurological abnormalities: torticollis (twisted neck), head tremors, circling, leg/wing paralysis",
            "Marked edema and swelling around the eyes, head, and neck",
            "Sudden, catastrophic drop in egg production; misshapen, thin-shelled, or depigmented eggs",
        ],
        "prevention": [
            "Mandatory vaccination: Hitchner B1 / LaSota live vaccines via eye-drop or drinking water followed by killed emulsion boosters",
            "Impose immediate farm quarantine upon any suspicion of clinical signs",
            "Install bird-proof netting to exclude wild birds, pigeons, and rodents from poultry houses",
            "Enforce strict biosecurity: vehicle spray gates, footbaths with glutaraldehyde or virucidal disinfectants",
            "Proper deep burial or sanitary incineration of bird carcasses with quicklime",
        ],
        "recommendation": (
            "CRITICAL BIOSECURITY ALERT: Isolate flock and lock down farm gates immediately. "
            "Report immediately to government veterinary authorities (Newcastle is an OIE/WOAH notifiable disease). "
            "There is no antiviral cure; strict quarantine, supportive electrolytes, and containment are imperative."
        ),
    },

    # =========================================================================
    # 4. SALMONELLOSIS (Active Model Prediction)
    # =========================================================================
    "Salmonella": {
        "name": "Salmonellosis (Pullorum / Fowl Typhoid / Paratyphoid)",
        "model_supported": True,
        "training_status": "Active Model Prediction (MobileNetV2 · 92.59% Precision, 83.33% Recall)",
        "severity": "High / Zoonotic Risk",
        "badge": "danger",
        "icon": "🧫",
        "scientific_name": "Salmonella enterica (serovars Pullorum, Gallinarum, Enteritidis, Typhimurium)",
        "pathogen_type": "Gram-negative Bacilli (Enterobacteriaceae)",
        "incubation": "1 – 3 days in chicks; variable in adults",
        "mortality": "High in chicks (up to 80%); causes chronic carrier states and reduced productivity in adults",
        "transmission": "Vertical (trans-ovarian transmission from breeder hens to eggs) and horizontal (contaminated feed, rodent droppings, water, litter, and human handlers).",
        "summary": (
            "Salmonellosis encompasses bacterial infections of major veterinary and public health concern. "
            "In chicks, it produces acute septicemia, somnolence, and chalky white diarrhea adhering to the vent ('pasty vent'). "
            "In commercial layers and broilers, Salmonella Enteritidis poses significant food safety risks via contaminated eggs and meat."
        ),
        "head_signs": "Somnolent appearance with eyes closed tightly; pale, shrunken, dry comb and wattles; drooping head huddled close to brooder.",
        "body_signs": "Ruffled plumage, reluctance to walk, swollen and tender hock joints, chalky white diarrhea encrusting cloacal feathers.",
        "fecal_signs": "Chalky white, yellowish-green, or mucoid diarrhea that cakes heavily around the vent feathers (pasty vent).",
        "symptoms": [
            "Chalky white, pasty diarrhea adhering to the vent feathers ('pasty vent')",
            "Somnolence: chicks stand huddled with closed eyes and drooping wings",
            "High early mortality within the first 2 – 3 weeks of chick brooding",
            "Swollen, warm, and painful hock joints resulting in lameness and reluctance to stand",
            "Severe dehydration, emaciation, and stunted flock uniformity",
        ],
        "prevention": [
            "Source day-old chicks exclusively from certified Salmonella-free parent and grandparent breeding flocks",
            "Conduct aggressive rodent, beetle, and fly eradication programs (rodents are primary reservoirs)",
            "Heat-pellet poultry feed (minimum 85°C) or treat mash with organic acid blends (formic/propionic acid)",
            "Perform regular drag-swab and cloacal swab surveillance cultures throughout rearing and laying cycles",
            "Strict worker hygiene: sanitizing handwash stations and dedicated footwear for hatcheries and brooding sheds",
        ],
        "recommendation": (
            "Isolate affected birds immediately. Submit samples to an accredited diagnostic laboratory for bacterial isolation "
            "and antimicrobial susceptibility testing. Administer veterinary-approved antibiotics with strict adherence to meat and egg withdrawal times."
        ),
    },

    # =========================================================================
    # 5. FOWL POX (Active Model Prediction)
    # =========================================================================
    "Fowl Pox": {
        "name": "Fowl Pox (Avian Pox)",
        "model_supported": True,
        "training_status": "Active Model Prediction (MobileNetV2 · 100.0% Precision, 100.0% Recall)",
        "severity": "Moderate to High",
        "badge": "warning",
        "icon": "🔴",
        "scientific_name": "Avipoxvirus (Poxviridae)",
        "pathogen_type": "Double-stranded Enveloped DNA Virus",
        "incubation": "4 – 14 days",
        "mortality": "Low in cutaneous (dry) form (< 5%); High in diphtheritic (wet) form (up to 50% due to suffocation)",
        "transmission": "Biting insects (mosquitoes), direct contact through broken skin/scratches, or inhalation of dried scab dust.",
        "summary": (
            "Fowl Pox is a slow-spreading viral infection manifesting in two distinct clinical forms: "
            "the Cutaneous (Dry) Form, marked by wart-like proliferative nodular lesions on unfeathered skin (comb, wattles, eyelids), "
            "and the Diphtheritic (Wet) Form, characterized by necrotic fibrinous plaques inside the mouth, pharynx, and trachea leading to asphyxiation."
        ),
        "head_signs": "Wart-like nodules progressing into yellowish blisters, followed by dark brown or black scabs on the comb, wattles, eyelids, and earlobes; swollen eyelids with eyes gummed shut.",
        "body_signs": "Occasional pox scabs on unfeathered legs and feet; ruffled feathers; emaciation; difficulty breathing with gasping in wet form.",
        "fecal_signs": "Droppings may appear loose or yellowish due to anorexia, but fecal lesions are non-pathognomonic.",
        "symptoms": [
            "Distinct wart-like nodular scabs on comb, wattles, face, and eyelids",
            "Swollen, crusty eyelids leading to temporary or permanent blindness",
            "Yellowish diphtheritic pseudomembranes in the oral cavity, larynx, and trachea (wet form)",
            "Severe respiratory distress, wheezing, and choking sounds",
            "Marked reduction in growth rates and steep drop in egg production",
        ],
        "prevention": [
            "Vaccinate birds using the wing-web stab method with live pigeon pox or fowl pox vaccine at 6 – 10 weeks of age",
            "Implement vector control: eliminate standing water, apply larvicides, and fog sheds against mosquitoes",
            "Avoid flock overcrowding and debeaking cannibalism to minimize skin injuries and scratches",
            "Quarantine new flock additions for at least 3 weeks before introducing them to the main poultry house",
            "Clean and disinfect drinking water systems regularly using chlorine or quaternary ammonium compounds",
        ],
        "recommendation": (
            "Isolate affected birds. Gently soften and treat dry cutaneous lesions with veterinary antiseptic ointment (e.g., iodine). "
            "For wet-form respiratory blockage, seek immediate veterinary assistance. Vaccinate unaffected healthy pens immediately to arrest spread."
        ),
    },

    # =========================================================================
    # 6. AVIAN INFLUENZA / BIRD FLU (Educational / Future Retraining)
    # =========================================================================
    "Avian Influenza": {
        "name": "Avian Influenza (Bird Flu / HPAI)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "Critical / Emergency / Zoonotic",
        "badge": "danger",
        "icon": "🚨",
        "scientific_name": "Influenza A virus (Orthomyxoviridae, H5N1 / H5N8 / H7N9)",
        "pathogen_type": "Enveloped Negative-Sense RNA Virus",
        "incubation": "Hours to 5 days",
        "mortality": "Near 100% within 24 – 48 hours in Highly Pathogenic Avian Influenza (HPAI)",
        "transmission": "Direct contact with infected wild waterfowl, secretions, feces, contaminated water bodies, fomites, and airborne dust.",
        "summary": (
            "Highly Pathogenic Avian Influenza (HPAI) is a catastrophic, high-impact zoonotic disease of international veterinary concern. "
            "It triggers widespread systemic hemorrhage, multi-organ necrosis, severe cyanosis of head structures, subcutaneous edema, "
            "and sudden death with few premonitory signs."
        ),
        "head_signs": "Severe cyanosis (deep purple or blackish discoloration) of comb and wattles; massive facial, comb, and wattle edema; hemorrhages on unfeathered skin; heavy nasal discharge.",
        "body_signs": "Petechial hemorrhages on shanks and feet; tremors; prostration; torticollis; sudden flock-wide death without prior warning.",
        "fecal_signs": "Watery, bright green or whitish diarrhea accompanied by severe dehydration.",
        "symptoms": [
            "Sudden, catastrophic flock mortality with birds found dead overnight",
            "Dark purple to blackish discoloration (cyanosis) of comb, wattles, and facial skin",
            "Extensive fluid swelling (edema) of the head, peri-orbital tissues, and neck",
            "Pinpoint bleeding (petechiae) and red patches on unfeathered shanks and feet",
            "Nasal discharge, coughing, sneezing, and complete cessation of egg production",
        ],
        "prevention": [
            "Establish absolute biosecurity: zero contact with wild migratory waterfowl or open outdoor water bodies",
            "Enforce full shower-in / shower-out facilities, tyre baths, and protective clothing for all farm personnel",
            "Treat all surface water before use; test poultry water supplies regularly",
            "Comply with national regulatory guidelines and reporting frameworks",
            "Strictly prohibit live-bird market visits by farm personnel",
        ],
        "recommendation": (
            "URGENT GOVERNMENT BIOSECURITY ALERT: Immediately seal and quarantine the farm. "
            "Do NOT move live birds, dead birds, or manure off premises. Contact national veterinary disease control authorities immediately. "
            "Exercise extreme personal protective precautions due to potential zoonotic transmission to humans."
        ),
    },

    # =========================================================================
    # 7. INFECTIOUS BRONCHITIS (Educational / Future Retraining)
    # =========================================================================
    "Infectious Bronchitis": {
        "name": "Infectious Bronchitis (IB)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "Moderate to High",
        "badge": "warning",
        "icon": "🫁",
        "scientific_name": "Avian coronavirus (Coronaviridae)",
        "pathogen_type": "Positive-sense Single-stranded RNA Coronavirus",
        "incubation": "18 – 36 hours",
        "mortality": "Up to 25% in young chicks (especially with nephropathogenic strains); low in adult layers but causes severe economic loss",
        "transmission": "Inhalation of airborne viral aerosols, contaminated dust, equipment, clothing, and water.",
        "summary": (
            "Infectious Bronchitis is an acute, highly contagious coronavirus infection of domestic chickens. "
            "It targets the respiratory tract, renal system (nephropathogenic strains), and female reproductive tract, "
            "resulting in severe tracheal rales, cystic oviducts, and permanently misshapen, wrinkled, thin-shelled eggs with watery albumen."
        ),
        "head_signs": "Swollen facial sinuses; clear to cloudy nasal discharge; wet eyes with conjunctivitis; open-mouth breathing with rattling.",
        "body_signs": "Huddling under brooders; wet droppings due to polyuria in nephropathogenic strains; severe decline in egg quality.",
        "fecal_signs": "Excessive wet, watery droppings with prominent white urates in nephropathogenic IB (damaged kidneys).",
        "symptoms": [
            "Severe respiratory rales: loud gasping, coughing, wheezing, and rattling sounds",
            "Coughing and watery nasal and ocular discharge in chicks",
            "Sharp 20% – 50% plunge in egg production in commercial layers",
            "Wrinkled, rough-surfaced, thin-shelled, or depigmented eggs with watery egg white",
            "Excessive water intake and wet droppings caused by renal tubule damage",
        ],
        "prevention": [
            "Administer live attenuated IB vaccines (H120 / Ma5) at day-old via coarse spray or eye-drop, followed by killed booster vaccines",
            "Match vaccination strains to regional circulating serotypes (e.g., Massachusetts, 4/91, QX)",
            "Maintain optimal house temperature and eliminate drafts during chick brooding",
            "Avoid high ammonia levels, which strip the protective cilia of the tracheal epithelium",
            "Implement thorough all-in/all-out poultry house disinfection between flocks",
        ],
        "recommendation": (
            "Provide supportive flock care: increase shed temperature by 2°C – 3°C, administer electrolyte and vitamin solutions in drinking water. "
            "Consult a poultry veterinarian to determine whether secondary bacterial complications require targeted antibiotic therapy."
        ),
    },

    # =========================================================================
    # 8. INFECTIOUS BURSAL DISEASE / GUMBORO (Educational / Future Retraining)
    # =========================================================================
    "Infectious Bursal Disease": {
        "name": "Infectious Bursal Disease (IBD / Gumboro)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "High / Immunosuppressive",
        "badge": "danger",
        "icon": "🛡️",
        "scientific_name": "Infectious bursal disease virus (Birnaviridae)",
        "pathogen_type": "Double-stranded Bi-segmented RNA Virus",
        "incubation": "2 – 3 days",
        "mortality": "20% – 60% in very virulent IBD (vvIBD) strains; higher morbidity in 3- to 6-week-old birds",
        "transmission": "Fecal-oral route through ingestion of contaminated feed, water, dust, equipment, and darkling beetles (Alphitobius diaperinus).",
        "summary": (
            "Gumboro disease selectively infects and destroys developing B-lymphocytes in the Bursa of Fabricius, "
            "causing severe, irreversible immunosuppression. Affected birds become highly vulnerable to secondary opportunistic infections "
            "and fail to mount antibody responses to other routine vaccines."
        ),
        "head_signs": "Depressed, somnolent appearance; birds sit with beak resting on litter; pale facial skin.",
        "body_signs": "Birds picking at their own cloacal vent; trembling and uncoordinated gait; severely ruffled plumage; prostration.",
        "fecal_signs": "Watery, whitish, chalky diarrhea containing high concentrations of urates; vent soiled with pasty white droppings.",
        "symptoms": [
            "Sudden spike in flock mortality peaking within 3 – 5 days and resolving by day 7",
            "Severe vent-pecking behavior directed at cloacal opening",
            "Watery, whitish-yellow diarrhea leading to dehydration and soiled feathers",
            "Extreme depression, trembling, anorexia, and refusal to move",
            "Marked swelling of the Bursa of Fabricius followed by rapid bursal atrophy",
        ],
        "prevention": [
            "Formulate hatchery vaccination using immune-complex (Icx) or recombinant vector vaccines (HVT-IBD)",
            "Administer intermediate or intermediate-plus live drinking water vaccines timed according to maternal antibody decay (Deventer formula)",
            "Eradicate darkling beetles and larvae, which harbor the birnavirus inside shed insulation and litter",
            "Thoroughly clean sheds using formaldehyde or glutaraldehyde-based disinfectants (the virus is highly resistant in environment)",
            "Ensure full resting/down-time periods (minimum 14 days) between production batches",
        ],
        "recommendation": (
            "Provide intensive supportive care: acidify drinking water, add electrolytes and vitamins C & E to counter dehydration and oxidative stress. "
            "Administer broad-spectrum coverage if secondary bacterial enteritis or E. coli infection develops."
        ),
    },

    # =========================================================================
    # 9. MAREK’S DISEASE (Educational / Future Retraining)
    # =========================================================================
    "Marek's Disease": {
        "name": "Marek’s Disease (Avian Leukosis / Range Paralysis)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "High / Oncogenic",
        "badge": "danger",
        "icon": "🦵",
        "scientific_name": "Gallid alphaherpesvirus 2 (Alphaherpesvirinae)",
        "pathogen_type": "Enveloped Double-stranded DNA Herpesvirus",
        "incubation": "3 – 4 weeks to several months",
        "mortality": "Up to 80% in unvaccinated flocks; persistent progressive paralysis",
        "transmission": "Inhalation of infectious poultry feather follicle dander and poultry house dust.",
        "summary": (
            "Marek's Disease is an oncogenic herpesvirus infection characterized by lymphocyte infiltration "
            "into peripheral nerve trunks (sciatic, brachial, vagus) leading to progressive paralysis, as well as visceral lymphoma "
            "tumors in liver, spleen, kidneys, and proventriculus, and ocular 'gray eye' lesions."
        ),
        "head_signs": "Grey or cloudy iris with irregular, pinpoint or split pupil ('ocular lymphomatosis' / 'gray eye'); loss of pupil accommodation; pale comb.",
        "body_signs": "Characteristic classical split-leg stance (one leg stretched forward, one leg stretched backward); wing droop; enlarged feather follicles; emaciation.",
        "fecal_signs": "Non-pathognomonic; chronic green-tinted loose droppings due to vagus nerve paralysis causing crop and gizzard impaction.",
        "symptoms": [
            "Classical paralysis posture: one leg extended forward and the other trailing backward",
            "Unilateral or bilateral wing drooping due to brachial nerve involvement",
            "Irregular, constricted pupil with greyish discoloration of the iris ('gray eye')",
            "Severe progressive emaciation and palpable muscle wasting around the keel bone",
            "Enlarged nodular feather follicle lesions ('skin leukosis') on dressed carcass",
        ],
        "prevention": [
            "Mandatory day-old subcutaneous vaccination or in ovo injection (day 18 of incubation) with HVT (FC-126) + Rispens (CVI988) vaccine",
            "Maintain strict hygiene during the first 3 weeks of chick life before vaccine immunity fully develops",
            "Filter shed air and remove feather dander dust via exhaust cleaning and HEPA filtration where feasible",
            "Select genetically resistant commercial chicken breeds and layer strains",
            "Never mix newly hatched chicks with older rearing birds producing feather dander",
        ],
        "recommendation": (
            "There is no medical cure or treatment for Marek's Disease. Affected paralyzed birds must be humanely culled to prevent suffering. "
            "Review hatchery vaccination records and ensure in ovo / day-old injection protocols are audited."
        ),
    },

    # =========================================================================
    # 10. FOWL CHOLERA (Educational / Future Retraining)
    # =========================================================================
    "Fowl Cholera": {
        "name": "Fowl Cholera (Avian Pasteurellosis)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "High / Acute Septicemia",
        "badge": "danger",
        "icon": "🧪",
        "scientific_name": "Pasteurella multocida",
        "pathogen_type": "Gram-negative Coccobacillus (Pasteurellaceae)",
        "incubation": "12 hours to 2 days",
        "mortality": "Very High in acute outbreaks (up to 70%); chronic forms cause localized joint and wattle abscesses",
        "transmission": "Secretions from mouth, nose, conjunctiva of infected birds, contaminated water, feed, rodents, wild birds, and animal bites.",
        "summary": (
            "Fowl Cholera is an infectious bacterial disease of chickens and turkeys. Acute outbreaks produce overwhelming bacteremia, "
            "cyanosis, and sudden flock deaths. Chronic forms produce localized fibrinous exudates in wattles, hock joints, and middle ear, "
            "causing characteristic swollen wattles and torticollis."
        ),
        "head_signs": "Severe, unilateral or bilateral swollen wattles containing thick cheesy pus; dark purplish cyanotic comb and wattles; mucus discharge from beak.",
        "body_signs": "Swollen hock and footpad joints; lameness; ruffled feathers; rapid breathing; sudden mortality of heavy birds in prime condition.",
        "fecal_signs": "Watery, yellowish or greenish diarrhea progressing rapidly before septicemic collapse.",
        "symptoms": [
            "Sudden high mortality with birds dying while on feeders or nesting boxes",
            "Marked, hard, fluid-filled swelling of the wattles ('swollen wattle disease')",
            "Dark red to purplish comb and facial cyanosis caused by septic shock",
            "Yellowish or greenish watery diarrhea accompanied by rapid breathing",
            "Swollen, warm hock joints and localized footpad abscesses causing lameness",
        ],
        "prevention": [
            "Administer inactivated oil-adjuvanted Pasteurella multocida bacterin vaccines (or live attenuated vaccines) at 8 and 12 weeks",
            "Rigorous rodent eradication: rats and mice carry and shed virulent Pasteurella strains in their saliva and feces",
            "Exclude wild carnivores, raccoons, dogs, cats, and waterfowl from poultry sheds",
            "Clean and sanitize water distribution lines with continuous water sanitizers (chlorine dioxide / sodium hypochlorite)",
            "Avoid introducing older birds of unknown disease status into the farm premises",
        ],
        "recommendation": (
            "Initiate immediate antibiotic treatment in drinking water (e.g., Oxytetracycline, Chlortetracycline, or Sulfamethazine) "
            "upon veterinary diagnosis. Surgically lance and drain chronic localized wattle abscesses under antiseptic conditions if advised."
        ),
    },

    # =========================================================================
    # 11. INFECTIOUS CORYZA (Educational / Future Retraining)
    # =========================================================================
    "Infectious Coryza": {
        "name": "Infectious Coryza (Avian Head Cold / Snot Cold)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "Moderate to High",
        "badge": "warning",
        "icon": "👃",
        "scientific_name": "Avibacterium paragallinarum",
        "pathogen_type": "Gram-negative Pleomorphic Rod (Pasteurellaceae)",
        "incubation": "1 – 3 days",
        "mortality": "Low (2% – 10%), but morbidity is extremely high (up to 90%) with severe egg production losses",
        "transmission": "Direct contact with infected carriers, inhalation of airborne droplets, and drinking water contaminated with nasal exudates.",
        "summary": (
            "Infectious Coryza is an acute respiratory bacterial infection of chickens characterized by acute catarrhal inflammation "
            "of the upper respiratory tract. It produces dramatic facial edema, swollen wattles, foul-smelling nasal discharge, "
            "and sharp declines in growth and egg output."
        ),
        "head_signs": "Pronounced facial edema and infraorbital sinus swelling ('owl head' appearance); eyes swollen shut by sticky exudate; foul-smelling (fetid) nasal discharge; swollen wattles.",
        "body_signs": "Ruffled feathers; reduced feed intake; weight loss; wet plumage around wings where the bird rubs its runny nose.",
        "fecal_signs": "Generally normal, but may soften during peak anorexia.",
        "symptoms": [
            "Severe facial swelling around the eyes and infraorbital sinuses ('owl-faced' birds)",
            "Thick, sticky, foul-smelling nasal and ocular discharge",
            "Adherent eyelids gummed closed with crusty, foamy exudate",
            "Severe drop in feed intake and 10% – 40% loss in commercial egg production",
            "Labored breathing, rales, sneezing, and snicking sounds heard across the house",
        ],
        "prevention": [
            "Vaccinate pullets with trivalent inactivated Coryza bacterin (containing serovars A, B, and C) twice before point-of-lay",
            "Practice strict all-in/all-out production; recovered birds remain life-long carriers and must never mix with clean pullets",
            "Sanitize drinking water continuously with 2 – 3 ppm free available chlorine to stop transmission through waterers",
            "Provide excellent cross-ventilation to remove airborne droplet nuclei and reduce shed humidity",
            "Quarantine any outside replacement stock for a minimum of 4 weeks",
        ],
        "recommendation": (
            "Administer veterinary-prescribed water-soluble antimicrobials (e.g., Erythromycin, Sulfadimethoxine, or Enrofloxacin). "
            "Flush and clean water lines daily. Flush encrusted eyes with sterile saline solution to restore bird vision and feeding."
        ),
    },

    # =========================================================================
    # 12. MYCOPLASMOSIS / CRD (Educational / Future Retraining)
    # =========================================================================
    "Mycoplasmosis": {
        "name": "Mycoplasmosis (Chronic Respiratory Disease / CRD)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "Moderate to High / Chronic",
        "badge": "warning",
        "icon": "😮",
        "scientific_name": "Mycoplasma gallisepticum (MG) / Mycoplasma synoviae (MS)",
        "pathogen_type": "Cell-wall-less Mollicute Bacterium",
        "incubation": "6 – 21 days",
        "mortality": "Low alone, but high (up to 30%) when complicated by E. coli or viral respiratory co-infections (Complicated CRD)",
        "transmission": "Vertical (trans-ovarian from infected breeder hens to eggs) and horizontal via aerosols, feed, water, and footwear.",
        "summary": (
            "Mycoplasmosis (CRD) is a persistent, slow-moving respiratory disease. Because Mycoplasmas lack a bacterial peptidoglycan cell wall, "
            "they are naturally resistant to beta-lactam antibiotics (penicillins/cephalosporins). When compounded by Newcastle, IB, or E. coli, "
            "it causes severe airsacculitis, perihepatitis, and carcass condemnation."
        ),
        "head_signs": "Foamy, frothy bubbles in the inner corner of the eye; swollen infraorbital sinuses; nasal discharge; coughing and head-shaking.",
        "body_signs": "Ruffled plumage; persistent wet coughing and tracheal gurgles; in MS infections, swollen hock joints and breast blisters.",
        "fecal_signs": "May exhibit greenish droppings in chronic debilitated stages.",
        "symptoms": [
            "Persistent snicking, coughing, tracheal rattling, and wheezing sounds",
            "Frothy, foamy bubbles at the corner of the eye (ocular discharge)",
            "Swelling of the infraorbital sinuses directly below the eyes",
            "Slow, insidious drop in egg production and poor feed conversion efficiency",
            "Joint swelling and lameness when caused by Mycoplasma synoviae (MS)",
        ],
        "prevention": [
            "Purchase chicks exclusively from certified MG/MS-clean parent breeding stock monitored by serum plate agglutination (SPA)",
            "Vaccinate pullets with live attenuated MG vaccines (ts-11, 6/85, or F-strain) or recombinant vector vaccines",
            "Maintain immaculate air quality: keep shed ammonia below 15 ppm and prevent dust accumulation",
            "Strict farm biosecurity: avoid mixing multi-age birds on the same production site",
            "Heat-treat hatching eggs or dip in antibiotic solutions at specialized hatcheries",
        ],
        "recommendation": (
            "Because Mycoplasma lacks a cell wall, use macrolides, pleuromutilins, or tetracyclines "
            "(e.g., Tylosin, Tiamulin, or Doxycycline) as prescribed by a poultry veterinarian. Observe all milk/meat/egg withdrawal regulations."
        ),
    },

    # =========================================================================
    # 13. COLIBACILLOSIS / E. COLI (Educational / Future Retraining)
    # =========================================================================
    "Colibacillosis": {
        "name": "Colibacillosis (Avian Pathogenic E. coli / APEC)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "High / Secondary Complication",
        "badge": "danger",
        "icon": "🦠",
        "scientific_name": "Escherichia coli (Avian Pathogenic strains: O1, O2, O78)",
        "pathogen_type": "Gram-negative Rod (Enterobacteriaceae)",
        "incubation": "1 – 3 days",
        "mortality": "5% – 25%; major contributor to carcass condemnation at processing plants",
        "transmission": "Inhalation of contaminated fecal dust, cracked eggshell penetration, navel infection in day-old chicks (omphalitis), contaminated water.",
        "summary": (
            "Colibacillosis is one of the most prevalent bacterial diseases of commercial poultry worldwide. "
            "It commonly acts as a secondary opportunistic pathogen following respiratory viral insults or immunosuppression, "
            "producing severe fibrinous airsacculitis, perihepatitis ('frosted liver'), pericarditis, and yolk sac infection (omphalitis)."
        ),
        "head_signs": "Dullness, closed eyes, drooping head, occasional swollen head syndrome (SHS) when combined with avian metapneumovirus.",
        "body_signs": "Severely ruffled feathers, huddling near heat sources, swollen unhealed navels in day-old chicks, emaciation.",
        "fecal_signs": "Yellowish, pasty or greenish-white diarrheal droppings.",
        "symptoms": [
            "Elevated early-chick mortality during the first 7 days due to omphalitis (unhealed navels)",
            "Severe respiratory distress, wheezing, and moist rales (airsacculitis)",
            "Unthrifty, lethargic birds huddled with ruffled plumage and reluctance to feed",
            "Swollen abdomen and ascites (fluid accumulation) in broilers",
            "Yellowish-white fecal staining around the cloaca",
        ],
        "prevention": [
            "Maintain impeccable hatchery hygiene: avoid setting floor eggs, disinfect setters/hatchers between batches",
            "Optimize ventilation to keep litter dust and airborne bacterial colony-forming units (CFUs) low",
            "Sanitize drinking water 24/7 (chlorination or acid-based water sanitizers)",
            "Control primary viral respiratory infections (ND, IB, IBD) that compromise respiratory mucosal barriers",
            "Ensure brooding temperature is accurate so chicks absorb yolk sacs promptly",
        ],
        "recommendation": (
            "Obtain antimicrobial susceptibility testing from an accredited lab (many APEC strains exhibit multi-drug resistance). "
            "Administer approved veterinary antibiotics. Correct shed ventilation, litter moisture, and water sanitation immediately."
        ),
    },

    # =========================================================================
    # 14. ASPERGILLOSIS (Educational / Future Retraining)
    # =========================================================================
    "Aspergillosis": {
        "name": "Aspergillosis (Brooder Pneumonia / Mycotic Pneumonia)",
        "model_supported": False,
        "training_status": "Educational Reference (Requires Additional Labeled Clinical Dataset for Retraining)",
        "severity": "Moderate to High / Fungal",
        "badge": "warning",
        "icon": "🍄",
        "scientific_name": "Aspergillus fumigatus / Aspergillus flavus",
        "pathogen_type": "Filamentous Environmental Mold (Fungus)",
        "incubation": "1 – 3 days",
        "mortality": "Up to 50% in young brooder chicks; chronic sporadic wasting in adult birds",
        "transmission": "Inhalation of massive numbers of airborne fungal conidiospores from moldy litter, contaminated straw/shavings, or moldy feed.",
        "summary": (
            "Aspergillosis (Brooder Pneumonia) is a non-contagious environmental fungal infection of the respiratory tract. "
            "Chicks inhale fungal spores that germinate inside lungs and thoracic air sacs, creating characteristic yellowish-white "
            "mycotic granulomas, gasping, and occasionally invading the central nervous system or eye chambers."
        ),
        "head_signs": "Open-beak gasping without any audible gurgling or respiratory rales; ocular cloudiness with yellowish cheesy plug in eye chamber.",
        "body_signs": "Rapid breathing, extended necks, somnolence, uncoordinated staggering or torticollis when fungus invades the brain, emaciation.",
        "fecal_signs": "Non-pathognomonic; may be loose or watery secondary to severe dehydration.",
        "symptoms": [
            "Silent, open-mouth gasping without mucous rattling or sneezing sounds",
            "Rapid, labored abdominal respiration (dyspnea)",
            "Yellowish cheesy plaques in the conjunctival sac of the eye leading to blindness",
            "Extreme lethargy, anorexia, and high mortality in 1- to 2-week-old chicks",
            "Occasional loss of equilibrium, head-tilting, and paralysis (cerebral aspergillosis)",
        ],
        "prevention": [
            "Never use damp, musty, or moldy wood shavings, straw, or peanut hulls for poultry bedding",
            "Store poultry feed in clean, moisture-proof silos; clean feed augers and bins to prevent caked moldy feed",
            "Sanitize hatchery incubators and ventilation ducts with enilconazole or antifungal fumigants",
            "Avoid excessive relative humidity in chick brooding rooms",
            "Discard any mold-contaminated feed bags immediately",
        ],
        "recommendation": (
            "There is no cost-effective or approved antifungal flock treatment for brooder pneumonia. "
            "Immediately identify and remove the moldy bedding or feed source. Replace litter with fresh, dry shavings. "
            "Apply antifungal misting (e.g., copper sulfate or enilconazole) in empty shed environments."
        ),
    },
}

# Support legacy alias 'New Castle Disease' pointing to canonical 'Newcastle Disease'
if "New Castle Disease" not in DISEASE_INFO and "Newcastle Disease" in DISEASE_INFO:
    DISEASE_INFO["New Castle Disease"] = DISEASE_INFO["Newcastle Disease"]

# Helper categories (preserving exact CLASS_NAMES order for active classes)
ACTIVE_MODEL_CLASSES = [c for c in CLASS_NAMES if c in DISEASE_INFO and DISEASE_INFO[c]["model_supported"]]
REFERENCE_CLASSES = [c for c, d in DISEASE_INFO.items() if not d["model_supported"] and c != "New Castle Disease"]
