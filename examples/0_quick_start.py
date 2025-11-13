from aethergraph import graph_fn, NodeContext 
from aethergraph import start_server 

# define a simple graph function -- simply an async function with the @graph_fn decorator
@graph_fn(name="hello_world")
async def hello_world(input_text: str, *, context: NodeContext) -> str:
    # log the function call by context's logger
    context.logger().info("hello_world function called")

    # send a greeting message to the default channel (Console). More channels can be configured. 
    await context.channel().send_text(f"👋 Hello! You sent: {input_text}")

    # use llm service to generate a response with context.llm()
    llm_text, _usage = await context.llm().chat(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": f"Say hi back to: {input_text}"},
        ],
    )

    await context.channel().send_text(f"🤖 LLM responded: {llm_text}")

    context.logger().info("hello_world function completed")
    return {"response": llm_text}

if __name__ == "__main__":
    from aethergraph.runner import run 

    # start the AetherGraph server before running any graph functions
    url = start_server() # default to localhost 8000 
    print(f"AetherGraph server started at: {url}")

    # run the hello_world graph function with some input
    result = run(hello_world, inputs={"input_text": "AetherGraph"})
    print(f"Graph function result: {result}")
