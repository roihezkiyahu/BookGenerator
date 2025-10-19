from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatLiteLLM
from typing import TypedDict, List, Dict, Any, Optional
from litellm import completion
from dotenv import load_dotenv
import os
from tools.agent_tools import AgentState
import json
from nodes.nodes import generate_story, validate_story, is_story_valid, validate_image_prompts, image_prompt_generator, is_valid_image_prompts, check_images_existance, merge_images, generate_images
from pprint import pprint
load_dotenv()

try:
    from langfuse.callback import CallbackHandler
    langfuse_handler = CallbackHandler()
except:
    langfuse_handler = None

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

# response = completion(
#     model="gemini/gemini-2.0-flash", 
#     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
# )

def build_graph(with_langfuse: bool = False):
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("generate_story", generate_story)
    graph_builder.add_node("validate_story", validate_story)
    graph_builder.add_node("validate_image_prompts", validate_image_prompts)
    graph_builder.add_node("image_prompt_generator", image_prompt_generator)
    graph_builder.add_node("merge_images", merge_images)
    graph_builder.add_node("generate_images", generate_images)

    graph_builder.add_edge(START, "generate_story")
    graph_builder.add_conditional_edges(
        "generate_story",
        is_story_valid,
        {
            "valid story": "image_prompt_generator",
            "invalid story": "validate_story"
        }
    )
    graph_builder.add_conditional_edges(
        "image_prompt_generator",
        is_valid_image_prompts,
        {
            "valid prompts, images exist": "merge_images",
            "valid prompts, images do not exist": "generate_images",
            "invalid image prompts": "validate_image_prompts"
        }
    )


    graph_builder.add_edge("generate_images", "merge_images")
    graph_builder.add_edge("merge_images", END)
    graph_builder.add_edge("validate_story", "generate_story")
    graph_builder.add_edge("validate_image_prompts", "image_prompt_generator")


    graph = graph_builder.compile().with_config({"callbacks": [langfuse_handler] if with_langfuse and langfuse_handler is not None else []})
    return graph

if __name__ == "__main__":
    graph = build_graph(with_langfuse=False)

    with open('output.png', 'wb') as f:
        f.write(Image(graph.get_graph().draw_mermaid_png()).data)

    response = graph.invoke(AgentState(title="Sharki the friendly shark",
                            number_of_pages=5,
                            story_description="A story about a friendly shark who makes friends with other sea creatures",
                                characters_metadata=[{"name": "Sharki", "description": "A friendly blue shark with a big smile who loves to make friends"}],
                                image_style="digital art style children's book",
                                use_imagen_3=False,
                                generated_images={}))

    # response = graph.invoke(AgentState(title="Geffen's underwater adventure",
    #                         number_of_pages=5,
    #                         story_description="A story about a 1 year old boy who goes on an adventure in the deep sea",
    #                             characters_metadata=[{"name": "Geffen", "description": "A 1 year old boy with a brown hair and a blue eyes"}],
    #                             image_style="digital art style children's book",
    #                             use_imagen_3=False,
    #                             generated_images={}))

    pprint(json.dumps(response, indent=4))

