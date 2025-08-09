# from agents import Agent, Runner, RunContextWrapper, GuardrailFunctionOutput, input_guardrail, output_guardrail, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool, handoff
# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel
# import asyncio

# load_dotenv()

# # -----------------------------------------
# # API KEY LOADING
# # -----------------------------------------
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     raise ValueError("GEMINI_API_KEY environment variable is not set.")

# # -----------------------------------------
# # MODEL SETUP
# # -----------------------------------------
# external_client = AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-1.5-flash",
#     openai_client=external_client
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True
# )

# # -----------------------------------------
# # Pydantic Models
# # -----------------------------------------
# class UserInfo(BaseModel):
#     name: str
#     account_number: str
#     balance: float
#     pin: int

# class OutputType(BaseModel):
#     response: str
    
# # class StructuredResponse(BaseModel):
# #     user_name: str
# #     account_no: str
# #     response: str    

# # -----------------------------------------
# # Agents
# # -----------------------------------------
# deposit_agent = Agent(
#     name="Deposit Agent",
#     instructions="You are a deposit agent. Answer the user's questions about making deposits. Start response as: 'Deposit Agent tool called'.",
     
# )

# withdrawal_agent = Agent(
#     name="Withdrawal Agent",
#     instructions="You are a withdrawal agent. Answer the user's questions about making withdrawals. Start response as: 'Withdrawal Agent tool called'.",
   
# )

# balance_agent = Agent(
#     name="Balance Agent",
#     instructions="You are a balance agent. Answer the user's questions about checking account balances. Start response as: 'Balance Agent tool called'.",
     
# )

# class input_check(BaseModel):
#     is_bank_related: bool

# # Guardrail Input Agent
# input_guardrail_agent = Agent(
#     name="Guardrail Agent",
#     instructions="You are a guardrail agent. Your job is to ensure that the user is asking about banking-related topics.",
#     output_type=input_check,
#     model=model,
# )

# @input_guardrail
# async def banking_guardrail(
#     ctx: RunContextWrapper[None], agent: Agent, input: str
# ) -> GuardrailFunctionOutput:
#     res = await Runner.run(input_guardrail_agent, input=input, context=ctx.context, run_config=config)
#     return GuardrailFunctionOutput(
#         output_info=res.final_output.is_bank_related,
#         tripwire_triggered=False,
#     )

# # Guardrail Output Agent
# output_guardrail_agent = Agent(
#     name="Guardrail Output Agent",
#     instructions="You are a guardrail output agent. Your job is to ensure that the output of the agent is appropriate and does not contain any sensitive information like pin or user's balance.",
#     # output_type=OutputType,  # ✅ model for guardrail
# )

# @output_guardrail
# async def output_guardrail_fn(
#     ctx: RunContextWrapper[None], agent: Agent, output: str
# ) -> GuardrailFunctionOutput:
#     res = await Runner.run(output_guardrail_agent, input=output, context=ctx.context, run_config=config)
#     return GuardrailFunctionOutput(
#         output_info=res.final_output,
#         tripwire_triggered=False,
#     )

# # -----------------------------------------
# # Tools
# # ---------------------------------------
# user_data = UserInfo(name="Maryam", account_number="987654321", balance=150000.0, pin=4321)
# @function_tool
# async def get_user_info(ctx: RunContextWrapper[UserInfo]) -> dict:
#     """Retrieve user information including name, account number, and balance."""
    
#     return {
#         "user_name": user_data.name,
#         "account_no": user_data.account_number,
#         "response": f"Your current balance is ${user_data.balance:.2f}"
#     }

# # -----------------------------------------
# # Main Agent
# # -----------------------------------------
# main_agent = Agent[user_data](
#     name="Bank Agent",
#     instructions="""
#     You are a bank agent. Answer the user's questions about banking.
#     Always handsoff according to the user's request.
#     If the user asks about deposits, handoff to the deposit agent.
#     If they ask about withdrawals, handoff to the withdrawal agent.
#     If they ask about account balances, handoff to the balance agent.
#     Use tools to get users information about name, account number, and balance.
#     If the user asks about interest rates, provide the current interest rate on savings accounts.
#     """,
#     model=model,
#     handoffs=[deposit_agent, withdrawal_agent, balance_agent],
#     tools=[get_user_info],
#     # tool_use_behavior="always",
#     input_guardrails=[banking_guardrail],
#     output_guardrails=[output_guardrail_fn],
# )

# # -----------------------------------------
# # Runner
# # -----------------------------------------
# async def main():
#     try:
#         result = await Runner.run(
#             main_agent,
#             input="tell recipe of cake.",
#             run_config=config
#         )
        
#         print(result.final_output)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     asyncio.run(main())

from agents import (
    Agent,
    Runner,
    RunContextWrapper,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    RunConfig,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    function_tool,
    handoff,
)
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import asyncio
import json

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

client = AsyncOpenAI(api_key=openai_api_key)

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# -----------------------------------------
# Pydantic Models
# -----------------------------------------
class UserInfo(BaseModel):
    name: str
    account_number: str
    balance: float
    pin: int


class OutputType(BaseModel):
    response: str


class InputCheck(BaseModel):
    is_bank_related: bool


# -----------------------------------------
# Agents (no output_type that returns Pydantic objects)
# -----------------------------------------
deposit_agent = Agent(
    name="Deposit Agent",
    instructions="""You are a deposit agent. Answer the user's questions about making deposits. 
    Return answers in plain text or JSON when appropriate.""",
    model=model,
)

withdrawal_agent = Agent(
    name="Withdrawal Agent",
    instructions="You are a withdrawal agent. Answer the user's questions about making withdrawals.",
    model=model,
)

balance_agent = Agent(
    name="Balance Agent",
    instructions="You are a balance agent. If asked for balance, call the get_user_info tool and return a short answer.",
    model=model,
)

# -----------------------------------------
# Guardrail Input Agent (returns InputCheck-like dict or model)
# -----------------------------------------
input_guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions=(
        """You are a guardrail. Given the user's text, return JSON with a boolean key 
        'is_bank_related' set to true if the user asks about banking/accounts/payments/etc., 
        otherwise false. Return only JSON, e.g. {\"is_bank_related\": true}."""
    ),
    output_type=InputCheck,
    model=model,
)


@input_guardrail
async def banking_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, user_input: str
) -> GuardrailFunctionOutput:
    """
    Robustly handle different shapes of res.final_output:
      - dict (preferred)
      - Pydantic model instance (has attribute)
      - JSON string
      - plain string
    """
    res = await Runner.run(
        input_guardrail_agent, input=user_input, context=ctx.context, run_config=config
    )

    final = getattr(res, "final_output", None)
    is_bank = False

    # 1) dict-like
    if isinstance(final, dict):
        is_bank = bool(final.get("is_bank_related", False))
    # 2) pydantic-like object or attribute
    elif final is not None and hasattr(final, "is_bank_related"):
        try:
            is_bank = bool(getattr(final, "is_bank_related"))
        except Exception:
            is_bank = False
    # 3) string (maybe JSON)
    elif isinstance(final, str):
        try:
            parsed = json.loads(final)
            if isinstance(parsed, dict):
                is_bank = bool(parsed.get("is_bank_related", False))
            else:
                is_bank = "true" in final.lower()
        except Exception:
            is_bank = "bank" in final.lower() or "balance" in final.lower()

    return GuardrailFunctionOutput(output_info=not is_bank, tripwire_triggered=True)


# -----------------------------------------
# Guardrail Output Agent
# -----------------------------------------
output_guardrail_agent = Agent(
    name="Guardrail Output Agent",
    instructions=(
        "You are a guardrail for outputs. Ensure the text does not leak sensitive info like PIN. "
        "If you detect sensitive details, respond with a safe refusal message. Otherwise return the original response."
    ),
    model=model,
)


@output_guardrail
async def output_guardrail_fn(
    ctx: RunContextWrapper[None], agent: Agent, output: str
) -> GuardrailFunctionOutput:
    res = await Runner.run(
        output_guardrail_agent, input=output, context=ctx.context, run_config=config
    )

    final = getattr(res, "final_output", None)
    # Normalize to string (most guardrail checks inspect text)
    safe_output = final
    if isinstance(final, dict):
        safe_output = json.dumps(final)
    elif final is None:
        safe_output = ""
    else:
        # already string or model -> str()
        safe_output = str(final)

    return GuardrailFunctionOutput(output_info=safe_output, tripwire_triggered=False)


# -----------------------------------------
# Tools
# -----------------------------------------
# static user_data for the demo
user_data = UserInfo(name="Maryam", account_number="987654321", balance=150000.0, pin=4321)


@function_tool
async def get_user_info(ctx: RunContextWrapper[None]) -> dict:
    """
    Return a plain dict (NOT a Pydantic instance) to avoid runtime `.extend`/type issues.
    """
    return {
        "user_name": user_data.name,
        "account_no": user_data.account_number,
        "response": f"Your current balance is ${user_data.balance:.2f}",
    }


# -----------------------------------------
# Main Agent
# -----------------------------------------
main_agent = Agent(
    name="Bank Agent",
    instructions="""
    You are a helpful bank agent. Follow these rules:
    - If the user asks about deposits -> handoff to Deposit Agent.
    - If the user asks about withdrawals -> handoff to Withdrawal Agent.
    - If the user asks about account balance or account details -> call the get_user_info tool.
    - Always return concise answers. If returning structured info, return JSON with keys:
      user_name, account_no, response.
    """,
    model=model,
    handoffs=[deposit_agent, withdrawal_agent, balance_agent],
    tools=[get_user_info],
      # ensure model uses the tool for account queries
    input_guardrails=[banking_guardrail],
    output_guardrails=[output_guardrail_fn],
)


# -----------------------------------------
# Runner
# -----------------------------------------
async def main():
    try:
        # Example 1: banking-related query -> guardrail will allow, tool will be used
        result = await Runner.run(
            main_agent,
            input="what is my current balance?",
            run_config=config,
        )

        print("== Result FINAL OUTPUT ==")
        print(result.final_output)  # final_output may be dict/string/etc.
        print("== Type of final_output ==", type(result.final_output))

        # Example 2: non-banking query -> guardrail should block or allow based on check
        result2 = await Runner.run(
            main_agent,
            input="i have to deposite but i forget my pin,plz help.",
            run_config=config,
        )
        print("== Result2 FINAL OUTPUT ==")
        print(result2.final_output)
        print("== Type ==", type(result2.final_output))

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
