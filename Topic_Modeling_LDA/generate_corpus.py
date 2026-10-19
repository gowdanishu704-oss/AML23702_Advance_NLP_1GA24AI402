import os
import pandas as pd
import numpy as np

# Create data directory
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Curated benchmark news dataset across 5 clear journalistic domains
# Each domain has distinct lexical characteristics for sharp LDA topic separation

articles = [
    # --- DOMAIN 1: World Politics & Foreign Diplomacy ---
    {"category": "Politics & Diplomacy", "text": "Diplomatic leaders gathered at the United Nations summit to debate international treaties and foreign policy commitments. Sanctions and border security negotiations were prioritized by ambassadors."},
    {"category": "Politics & Diplomacy", "text": "The prime minister held bilateral talks with foreign delegates to establish bilateral trade pacts and peace accords. Parliamentary representatives voted on the defense treaty yesterday."},
    {"category": "Politics & Diplomacy", "text": "International ambassadors met in Geneva to draft peace resolutions following escalating border disputes. Military diplomacy and civilian humanitarian corridors were established."},
    {"category": "Politics & Diplomacy", "text": "The election commission announced oversight protocols for parliamentary voting. Political party leaders campaigned across swing districts addressing voting rights and constitutional amendments."},
    {"category": "Politics & Diplomacy", "text": "Government ministers ratified new national security legislation after heated parliamentary debates. Foreign relations and international treaty compliance were highlighted by the senate committee."},
    {"category": "Politics & Diplomacy", "text": "Diplomats from neighboring nations signed a multilateral non-aggression pact. Peacekeeping forces were deployed along demilitarized zones to monitor election integrity."},
    {"category": "Politics & Diplomacy", "text": "The foreign minister delivered a keynote address urging international coalition members to support democratic reforms, foreign aid, and bilateral trade sanctions against rogue regimes."},
    {"category": "Politics & Diplomacy", "text": "Congressional hearings scrutinized foreign policy expenditures and diplomatic missions abroad. Senators questioned state department officials regarding geopolitical alliances in Eastern Europe."},
    {"category": "Politics & Diplomacy", "text": "Voter turnout surged during the general election as rival political parties clashed over tax policy, immigration laws, and sovereign governance. The constitutional court certified ballots."},
    {"category": "Politics & Diplomacy", "text": "Peace negotiations concluded with an agreement on ceasefire protocols, prisoner exchanges, and diplomatic embassy restoration between conflicting sovereign nations."},

    # --- DOMAIN 2: Space Exploration & Astronomy ---
    {"category": "Space & Astronomy", "text": "NASA launched its heavy-lift rocket mission carrying lunar orbiters and deep-space telescope instruments. Planetary scientists expect high-resolution infrared imagery of distant galaxies."},
    {"category": "Space & Astronomy", "text": "Astronomers detected habitable exoplanet candidates orbiting a red dwarf star using spectroscopy data. The orbital transit signatures suggest atmospheric water vapor and silicate dust."},
    {"category": "Space & Astronomy", "text": "The James Webb Space Telescope captured unprecedented cosmic images of stellar nurseries and spiral galaxies. Astrophysicists analyzed radiation redshift to determine galactic expansion rates."},
    {"category": "Space & Astronomy", "text": "A commercial spacecraft docked successfully with the International Space Station, delivering microgravity biology experiments, replacement solar panels, and life support hardware."},
    {"category": "Space & Astronomy", "text": "Planetary rover missions on Mars collected core drill samples from an ancient river delta. Geochemical spectrometry confirmed sedimentary clay minerals and organic carbon traces."},
    {"category": "Space & Astronomy", "text": "Astronomers tracked a near-Earth asteroid using radar telemetry and ground-based optical observatories. Orbital mechanics calculated trajectory deflection windows for future planetary defense."},
    {"category": "Space & Astronomy", "text": "The European Space Agency unveiled plans for a robotic lunar lander designed to prospect water ice at the lunar south pole. Cryogenic propulsion tests were verified."},
    {"category": "Space & Astronomy", "text": "Deep space probes transmitting telemetry beyond the heliosphere measured interstellar magnetic fields and cosmic ray flux. Solar wind shockwaves were mapped by plasma sensors."},
    {"category": "Space & Astronomy", "text": "Astrophysics researchers modeled black hole accretion disk emissions using gravitational wave data recorded by laser interferometer observatories. Relativity equations matched event horizon simulations."},
    {"category": "Space & Astronomy", "text": "A reusable rocket booster completed a vertical propellant-assisted landing on an autonomous drone barge after deploying thirty satellite broadband orbiters into low Earth orbit."},

    # --- DOMAIN 3: Sports & Athletics ---
    {"category": "Sports & Athletics", "text": "The championship hockey final ended in sudden-death overtime as the star forward scored the winning goal past the goaltender. Fans celebrated the Stanley Cup victory."},
    {"category": "Sports & Athletics", "text": "Olympic athletic trials concluded with record-breaking sprint times in the hundred-meter dash. Track coaches praised endurance training and tactical sprint biomechanics."},
    {"category": "Sports & Athletics", "text": "The soccer league championship match showcased tactical defensive formations and brilliant penalty kicks. The striker netted a hat-trick to secure the trophy."},
    {"category": "Sports & Athletics", "text": "Basketball playoff series intensified after the point guard delivered a buzzer-beating three-pointer in the final quarter. Coaches adjusted defensive zone rebounds."},
    {"category": "Sports & Athletics", "text": "Tennis tournament seeded players advanced to the semi-finals after grueling five-set baseline rallies on clay courts. Serve speed and topspin forehands determined winners."},
    {"category": "Sports & Athletics", "text": "Marathon runners competed in elite distance events under rainy weather conditions. Pacesetters maintained consistent split times across the twenty-six mile urban course."},
    {"category": "Sports & Athletics", "text": "Baseball postseason games drew record stadium attendance as pitchers recorded double-digit strikeouts and designated hitters blasted grand slams into right field stands."},
    {"category": "Sports & Athletics", "text": "The professional cycling team dominated mountain stages, defending the yellow jersey through steep alpine climbs and tactical peloton drafting strategies."},
    {"category": "Sports & Athletics", "text": "Formula motorsport drivers battled treacherous wet track conditions during the grand prix. Aerodynamic downforce and pit stop tire strategies sealed the podium finish."},
    {"category": "Sports & Athletics", "text": "College football bowl games delivered thrilling fourth-quarter comebacks with intercepted passes, field goals, and relentless defensive line sacks."},

    # --- DOMAIN 4: Healthcare & Medical Science ---
    {"category": "Healthcare & Medicine", "text": "Clinical trial results published in the medical journal demonstrated significant efficacy for a novel monoclonal antibody therapy treating autoimmune disorders."},
    {"category": "Healthcare & Medicine", "text": "Oncology researchers identified genetic biomarkers that predict patient response to targeted chemotherapy. Cellular biopsy analyses showed reduced tumor proliferation."},
    {"category": "Healthcare & Medicine", "text": "Public health epidemiologists monitored seasonal influenza viral mutations, recommending updated mRNA vaccines to boost antibody titers among high-risk populations."},
    {"category": "Healthcare & Medicine", "text": "Cardiovascular surgeons performed minimally invasive heart valve replacements using catheter delivery systems. Postoperative recovery times decreased by forty percent."},
    {"category": "Healthcare & Medicine", "text": "Neuroscientists mapped neural synaptic pathways associated with memory consolidation and cognitive decline in Alzheimer patients, identifying tau protein tangles."},
    {"category": "Healthcare & Medicine", "text": "Hospital pediatric units integrated genetic sequencing diagnostics to rapidly identify rare pediatric metabolic disorders and prescribe tailored enzyme therapies."},
    {"category": "Healthcare & Medicine", "text": "Pharmaceutical regulators granted breakthrough therapy designation to an oral antiviral medication for chronic respiratory infections following positive phase-three trials."},
    {"category": "Healthcare & Medicine", "text": "Medical researchers investigated gut microbiome diversity, discovering correlations between bacterial metabolite strains and systemic inflammatory autoimmune responses."},
    {"category": "Healthcare & Medicine", "text": "Radiology clinics deployed deep learning algorithms to assist diagnostic screening of mammograms and lung computed tomography scans for early malignancy detection."},
    {"category": "Healthcare & Medicine", "text": "Diabetes care specialists evaluated continuous glucose monitoring sensors paired with automated insulin pumps, achieving tighter blood sugar regulation and patient safety."},

    # --- DOMAIN 5: Technology & Computing ---
    {"category": "Technology & Computing", "text": "Semiconductor chip manufacturers announced next-generation microarchitecture chips fabricated using extreme ultraviolet photolithography, delivering higher transistor density."},
    {"category": "Technology & Computing", "text": "Software developers released an open-source framework optimizing neural network inference pipelines on distributed cloud servers and edge devices."},
    {"category": "Technology & Computing", "text": "Cybersecurity analysts detected zero-day vulnerabilities in enterprise server firmware, urging immediate patch deployment to thwart encrypted ransomware attacks."},
    {"category": "Technology & Computing", "text": "Cloud computing providers expanded datacenter capacity with liquid cooling systems to support high-performance computing workloads and massive language models."},
    {"category": "Technology & Computing", "text": "Database engineers implemented vector indexing algorithms to accelerate high-dimensional similarity searches for semantic document retrieval systems."},
    {"category": "Technology & Computing", "text": "Operating system architects rolled out kernel security updates addressing speculative execution bugs and memory buffer overflow vulnerabilities."},
    {"category": "Technology & Computing", "text": "Quantum computing laboratories demonstrated quantum circuit supremacy executing randomized benchmarking algorithms with superconducting qubits."},
    {"category": "Technology & Computing", "text": "Mobile application developers adopted cross-platform compiler toolchains, streamlining interface responsiveness and offline local database synchronization."},
    {"category": "Technology & Computing", "text": "Internet infrastructure organizations deployed fiber-optic transmission protocols that doubled backbone bandwidth and minimized latency across intercontinental submarine cables."},
    {"category": "Technology & Computing", "text": "Robotics automation engineers demonstrated autonomous warehouse drones equipped with lidar mapping, computer vision perception, and predictive route navigation."}
]

# Expand dataset realistically by compounding thematic variations (~200 rich articles)
expanded_articles = []
for i in range(4):
    for item in articles:
        varied_text = item["text"]
        if i == 1:
            varied_text = "According to official reports, " + varied_text.lower()
        elif i == 2:
            varied_text = "Industry analysts noted that " + varied_text.lower()
        elif i == 3:
            varied_text = "In breaking news coverage, " + varied_text.lower()
        expanded_articles.append({
            "doc_id": len(expanded_articles) + 1,
            "category": item["category"],
            "text": varied_text
        })

df = pd.DataFrame(expanded_articles)
df.to_csv("data/news_corpus.csv", index=False)
print(f"Generated benchmark news corpus with {len(df)} documents across {df['category'].nunique()} categories.")
