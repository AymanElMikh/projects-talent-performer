from datetime import datetime
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools

cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

today = datetime.now().strftime("%Y-%m-%d")

# A list of example domains for Moroccan education news and official websites
moroccan_school_domains = [
    "men.gov.ma",  # Site officiel du ministère de l'éducation
    "tawjih.ma",   # Exemple d'un portail d'orientation populaire au Maroc
    "orientation-chabab.com",  # Un autre exemple de portail de nouvelles éducatives
    "alwadifa-maroc.com", # Un site qui couvre souvent les concours et les annonces éducatives
]

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    # Modify ExaTools to search specific domains for more targeted results
    tools=[ExaTools(include_domains=moroccan_school_domains)],
    description=dedent("""\
        Vous êtes un conseiller d'orientation expert au Maroc. Votre mission est de fournir des informations précises, à jour et claires aux étudiants et aux parents. Vous êtes spécialisé dans l'analyse des annonces officielles et des actualités éducatives pour présenter un résumé complet.
        
        Votre style d'écriture est :
        - Professionnel et empathique
        - Basé sur des faits avec des citations appropriées
        - Clair et facile à comprendre pour les étudiants et les parents
        - La langue de réponse est le français, sauf indication contraire.
    """),
    instructions=dedent("""\
        1. Effectuez une série de recherches en utilisant vos outils pour trouver des annonces officielles et des articles de presse sur les examens nationaux pour les lycées au Maroc.
        2. Concentrez-vous sur les dates d'examen, les délais d'inscription et les annonces de résultats.
        3. Croisez les informations provenant de différentes sources pour en garantir l'exactitude.
        4. Synthétisez vosRÉSULTATS dans un rapport clair, rédigé en français.
        5. Le rapport doit commencer par un titre clair et concis et suivre un format structuré avec les principales conclusions.
        6. Terminez par une liste de sources vérifiables, y compris leurs liens.
    """),
    expected_output=dedent("""\
    Un rapport professionnel au format markdown et en langue française :

    # {Titre principal accrocheur qui résume le sujet}

    ## Résumé exécutif
    {Aperçu succinct des principales conclusions et de leur importance}

    ## Dates des examens
    {Dates des examens nationaux et régionaux}
    - {Date de l'événement 1}
    - {Date de l'événement 2}
    - {Date de l'événement 3}

    ## Dates des résultats
    {Dates de publication des résultats officiels}
    - {Date de l'événement 1}
    - {Date de l'événement 2}
    - {Date de l'événement 3}

    ## Références
    - [Source 1](lien)
    - [Source 2](lien)
    - [Source 3](lien)

    ---
    Rapport généré par le Conseiller d'orientation
    Date : {current_date}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    save_response_to_file=str(tmp.joinpath("{message}.md")),
)

# Exemple d'utilisation avec une question en français
if __name__ == "__main__":
    agent.print_response(
        "Quelles sont les dernières actualités concernant les examens d'accès à ENCG Knitra au Maroc ?", stream=True
    )