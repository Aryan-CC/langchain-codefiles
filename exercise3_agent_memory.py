# exercise3_agent_memory.py
# Exercise 3: Agent Assembly & Memory

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from langchain import hub
from exercise2_retrieval import InvoiceSearchTool, InvoiceRetrievalQA
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Task 3: Instrument Callbacks for Observability
class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring agent execution"""
    
    def __init__(self, log_to_file=True):
        self.log_to_file = log_to_file
        self.execution_log = []
    
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Called when agent takes an action"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_action",
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log
        }
        
        self.execution_log.append(log_entry)
        print(f"🔧 Agent Action: {action.tool}")
        print(f"   Input: {action.tool_input}")
        print(f"   Reasoning: {action.log}\n")
        
        if self.log_to_file:
            self._write_to_log(log_entry)
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Called when agent finishes execution"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_finish",
            "output": finish.return_values,
            "log": finish.log
        }
        
        self.execution_log.append(log_entry)
        print(f"✅ Agent Finished")
        print(f"   Output: {finish.return_values}")
        print(f"   Final reasoning: {finish.log}\n")
        
        if self.log_to_file:
            self._write_to_log(log_entry)
    
    def on_tool_start(self, serialized, input_str: str, **kwargs):
        """Called when a tool starts execution"""
        print(f"🚀 Tool Starting: {serialized.get('name', 'Unknown')}")
        print(f"   Input: {input_str}")
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when a tool finishes execution"""
        print(f"🏁 Tool Output: {output[:200]}{'...' if len(output) > 200 else ''}\n")
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called when a tool encounters an error"""
        print(f"❌ Tool Error: {str(error)}\n")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts processing"""
        print(f"🧠 LLM Processing...")
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes processing"""
        print(f"💭 LLM Response received\n")
    
    def _write_to_log(self, log_entry):
        """Write log entry to file"""
        try:
            with open("agent_execution.log", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")

# Mock OCR Tool for demonstration
class OCRTool:
    """Mock OCR tool for document processing"""
    
    def extract_text(self, image_path: str) -> str:
        """Mock OCR text extraction"""
        # In a real implementation, this would use Azure Computer Vision or similar
        return f"Mock OCR result for {image_path}: This is extracted text from the document."
    
    def get_langchain_tool(self):
        """Return LangChain Tool object"""
        return Tool(
            name="ocr_extract",
            description="Extract text from images or documents using OCR. Use this when you need to read text from uploaded images or scanned documents.",
            func=self.extract_text
        )

# Task 1: Choose & Configure an Agent
class InvoiceAgent:
    """Structured agent with tool access and memory"""
    
    def __init__(self, enable_memory=True, enable_callbacks=True):
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.1
        )
        
        # Initialize tools
        self.search_tool = InvoiceSearchTool()
        self.retrieval_qa = InvoiceRetrievalQA()
        self.ocr_tool = OCRTool()
        
        # Create tools list
        self.tools = [
            self.search_tool.get_langchain_tool(),
            Tool(
                name="qa_retrieval",
                description="Answer detailed questions about invoices using retrieval and AI analysis. Use this for complex questions that need contextual understanding and reasoning.",
                func=self._qa_tool_wrapper
            ),
            self.ocr_tool.get_langchain_tool()
        ]
        
        # Task 2: Add Conversation Memory
        self.memory = None
        if enable_memory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        
        # Task 3: Setup callbacks
        self.callbacks = []
        if enable_callbacks:
            self.callback_handler = CustomCallbackHandler()
            self.callbacks = [self.callback_handler]
        
        # Get the ReAct prompt template
        try:
            self.prompt = hub.pull("hwchase17/react")
        except:
            # Fallback prompt if hub is not accessible
            from langchain.prompts import PromptTemplate
            self.prompt = PromptTemplate(
                input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
                template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
            )
        
        # Create the ReAct agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=self.callbacks,
            max_iterations=10
        )
    
    def _qa_tool_wrapper(self, query: str) -> str:
        """Wrapper for RetrievalQA tool"""
        result = self.retrieval_qa.query(query)
        return result["answer"]
    
    def chat(self, message: str):
        """Chat with the agent"""
        try:
            response = self.agent_executor.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_memory_summary(self):
        """Get conversation history summary"""
        if self.memory:
            return self.memory.chat_memory.messages
        return []
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            print("Memory cleared.")
    
    def get_execution_log(self):
        """Get execution log from callback handler"""
        if hasattr(self, 'callback_handler'):
            return self.callback_handler.execution_log
        return []

def test_exercise3():
    """Test Exercise 3 components"""
    print("=== Testing Exercise 3: Agent Assembly & Memory ===\n")
    
    # Initialize agent with all features
    agent = InvoiceAgent(enable_memory=True, enable_callbacks=True)
    
    # Test conversation with memory
    print("Testing multi-turn conversation with memory:")
    
    # First interaction
    response1 = agent.chat("Find invoices for Alice Johnson")
    print(f"User: Find invoices for Alice Johnson")
    print(f"Agent: {response1}\n")
    
    # Second interaction (should remember Alice Johnson)
    response2 = agent.chat("What was the total amount for her invoice?")
    print(f"User: What was the total amount for her invoice?")
    print(f"Agent: {response2}\n")
    
    # Third interaction (test OCR tool)
    response3 = agent.chat("Can you extract text from image.png?")
    print(f"User: Can you extract text from image.png?")
    print(f"Agent: {response3}\n")
    
    # Display memory contents
    print("=== Memory Contents ===")
    memory_messages = agent.get_memory_summary()
    for i, message in enumerate(memory_messages):
        print(f"{i+1}. {message.type}: {message.content[:100]}...")
    
    # Display execution log
    print("\n=== Execution Log Summary ===")
    execution_log = agent.get_execution_log()
    print(f"Total logged events: {len(execution_log)}")
    
    # Test memory clearing
    print(f"\nClearing memory...")
    agent.clear_memory()

if __name__ == "__main__":
    test_exercise3()