from agno.agent import Agent
from agno.embedder.ollama import OllamaEmbedder
from agno.knowledge.url import UrlKnowledge
from agno.models.ollama import Ollama
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType

knowledge = UrlKnowledge(
    urls=["https://docs.agno.com/introduction.md"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="agno_docs",
        search_type=SearchType.hybrid,
        # Use a local Ollama model for embeddings
        embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
    ),
)

# Store agent sessions in a SQLite database
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")

agent = Agent(
    name="Agno Assist",
    # Use a local Ollama model as the main agent model
    model=Ollama(id="mistral"),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    markdown=True,
)

if __name__ == "__main__":
    agent.knowledge.load(recreate=False)
    agent.print_response("What using Agno?", stream=True)