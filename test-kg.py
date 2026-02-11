import sys
from pathlib import Path

import asyncio

root = Path(__file__).parent.parent
sys.path.append(str(root))

from rag.knowledge_graph import KnowledgeGraph
from app_mcp.tools.knowledgegraph_tools import add_information_to_knowledge_graph
from core.agent import updation_knowledge_graph
from core.state import State
from config.settings import MEMORY_DB, DEFAULT_THREAD_ID


async def reinit_and_test():
    try:
        # # 1. Clear existing bad vectors

        # kg = KnowledgeGraph()

        # # 2. Define Aligned Strategic Nodes
        # nodes = [
        #     (
        #         "VIT Chennai",
        #         "Organization",
        #         "technical university research AI",
        #         "Leading technical university in Tamil Nadu.",
        #     ),
        #     (
        #         "DeepShield",
        #         "Project",
        #         "deepfake detection vision transformer hackhub",
        #         "Deepfake detection system using ViT.",
        #     ),
        #     (
        #         "Agriculture AI",
        #         "Project",
        #         "crop fertilizer recommendation maharashtra",
        #         "Crop recommendation system for Maharashtra.",
        #     ),
        #     (
        #         "Skin Disease AI",
        #         "Project",
        #         "dermatology classification hybrid cnn vit",
        #         "Hybrid CNN-ViT dermatological tool.",
        #     ),
        #     (
        #         "Vision Transformer",
        #         "Tool",
        #         "ViT architecture deep learning patches",
        #         "Architecture processing images as sequences.",
        #     ),
        #     (
        #         "CNN",
        #         "Tool",
        #         "convolutional neural network features",
        #         "Standard architecture for image features.",
        #     ),
        #     (
        #         "Yadeesh",
        #         "Person",
        #         "CSE student developer researcher",
        #         "Second-year student; lead developer.",
        #     ),
        #     (
        #         "Adam Karvel",
        #         "Person",
        #         "external collaborator google research",
        #         "Research partner for skin disease project.",
        #     ),
        #     (
        #         "Google",
        #         "Organization",
        #         "tech company transformer developer",
        #         "Company that pioneered transformers.",
        #     ),
        #     (
        #         "HackHub 25",
        #         "Event",
        #         "coding hackathon final stage",
        #         "Innovation event finalist.",
        #     ),
        # ]

        # for node_id, n_type, keywords, full_desc in nodes:
        #     kg.add_entity(node_id, n_type, keywords, full_desc)

        # # 3. Add Relationships
        # relationships = [
        #     ("Yadeesh", "VIT Chennai", "STUDIES_AT"),
        #     ("DeepShield", "VIT Chennai", "DEVELOPED_AT"),
        #     ("DeepShield", "Vision Transformer", "USES_MODEL"),
        #     ("Skin Disease AI", "Vision Transformer", "USES_MODEL"),
        #     ("Yadeesh", "DeepShield", "LEAD_DEVELOPER"),
        #     ("Adam Karvel", "Yadeesh", "COLLABORATES_WITH"),
        #     ("DeepShield", "HackHub 25", "FINALIST_AT"),
        # ]
        # for source, target, rel in relationships:
        #     kg.add_relationship(source, target, rel)

        # print("✅ Subgraph re-created with aligned embeddings.")

        # # 4. Run the Chat Logic Test
        # test_chat_history = """
        # human: I've decided to migrate the DeepShield vector storage to KuzuDB for better local performance.
        # clarification_agent: CLARIFICATION NEEDED: Will you continue using the Vision Transformer (ViT) architecture?
        # human: I'm sticking with the Vision Transformer for now because it worked well during the HackHub'25 finals.
        # """

        # print("\n--- Running Extraction & Similarity Search ---")
        # extracted = kg.generate_entity_relation(test_chat_history)
        # candidates = extracted.get("candidates", {})

        # similar_nodes = kg.search_similar_node(candidates.get("entities", []))
        # print("Similar Nodes Found:\n", similar_nodes)

        # test_state: State = {
        #     "messages": [],
        #     "summary": "",
        #     "last_knowledgegraph_timestamp": 1770195927.8211298,
        #     "next": "",
        # }

        # # 2. Pass the instance, not the class
        # await updation_knowledge_graph(
        #     state=test_state, thread_id=DEFAULT_THREAD_ID, db_path=MEMORY_DB
        # )

        # print("✅ Knowledge graph update successful.")
        # kg = KnowledgeGraph()
        # test_id = "Machine Vision 1"

        # updates = {
        #     "type": "Educational Institute",
        #     "description": "The description has been updated to include AI research focus.",
        # }

        # kg.modify_entity(node_id=test_id, updates=updates)

        # # Step 3: Retrieve from DB to verify persistence
        # # We use a raw query to check the exact fields
        # verify_query = (
        #     f"MATCH (n:Entity) WHERE n.id = '{test_id}' RETURN n.type, n.description"
        # )

        # # Kuzu returns a results object; converting to a dataframe makes it easy to read
        # results_df = kg.conn.execute(verify_query).get_as_df()

        # if not results_df.empty:
        #     updated_type = results_df.iloc[0]["n.type"]
        #     updated_desc = results_df.iloc[0]["n.description"]

        #     print(f"Database reflects -> Type: {updated_type}")
        #     print(f"Database reflects -> Desc: {updated_desc}")

        #     if (
        #         updated_type == "Educational Institute"
        #         and "AI research" in updated_desc
        #     ):
        #         print("✅ SUCCESS: Entity modified correctly.")
        #     else:
        #         print("❌ FAILURE: Data mismatch after modification.")
        # else:
        #     print("❌ FAILURE: Could not find the node after update.")
        #         df = await add_information_to_knowledge_graph(
        #             """I’ve stored the information in our current conversation context, so I can still refer to it while we chat:

        # - **Friend**: Kishore
        # - **Email**: kishoreag22@gmail.com"""
        #         )
        # print("KG Update Result:", df)

        kg = KnowledgeGraph()
        kg.visualize()
    except Exception as e:
        print(f"Test Failed: {e}")


if __name__ == "__main__":
    asyncio.run(reinit_and_test())
