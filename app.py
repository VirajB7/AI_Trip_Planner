import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser
import requests
from datetime import datetime, timedelta
import platform
from langsmith import traceable
from langchain_core.tracers.context import tracing_v2_enabled
import re

# Import LangGraph
from langgraph.graph import StateGraph, END
from typing import Dict, Any, TypedDict

# Import cost calculator
from cost_calculator import TravelCostCalculator

# -----------------------------
# 1️⃣ Define State for LangGraph
# -----------------------------
class TravelPlanningState(TypedDict):
    destination: str
    budget: str
    start_date: datetime
    num_days: int
    interests: list
    weather_validated: bool
    weather_message: str
    weather_details: list
    full_itinerary: str
    itinerary_generated: bool
    cost_result: Dict[str, Any]
    current_step: str
    error: str

# -----------------------------
# 2️⃣ Setup & Configuration
# -----------------------------
load_dotenv()

# Configure LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-travel-planner"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")

st.title("🌍 AI-Powered Travel Planner")
st.caption("Built with LangChain Runnables + DuckDuckGo Search + Gemini model + Weather Check + LangSmith Tracing + LangGraph")

# Initialize session state
if 'weather_validated' not in st.session_state:
    st.session_state.weather_validated = False
if 'weather_message' not in st.session_state:
    st.session_state.weather_message = ""
if 'itinerary_generated' not in st.session_state:
    st.session_state.itinerary_generated = False
if 'full_itinerary' not in st.session_state:
    st.session_state.full_itinerary = ""
if 'graph_state' not in st.session_state:
    st.session_state.graph_state = None

# -----------------------------
# 3️⃣ Create LangGraph Workflow
# -----------------------------
def create_travel_planner_graph():
    """Create the complete travel planning workflow using LangGraph"""
    
    workflow = StateGraph(TravelPlanningState)
    
    # -----------------------------
    # Weather Check Node
    # -----------------------------
    def weather_check_node(state: TravelPlanningState) -> TravelPlanningState:
        """Check weather conditions for the destination"""
        try:
            is_favorable, message, weather_details = check_weather_conditions(
                state["destination"], 
                state["start_date"], 
                state["num_days"]
            )
            
            return {
                **state,
                "weather_validated": is_favorable,
                "weather_message": message,
                "weather_details": weather_details,
                "current_step": "weather_checked"
            }
        except Exception as e:
            return {
                **state,
                "weather_validated": False,
                "weather_message": f"Weather check failed: {str(e)}",
                "current_step": "weather_error",
                "error": str(e)
            }
    
    # -----------------------------
    # Itinerary Generation Node
    # -----------------------------
    def itinerary_generation_node(state: TravelPlanningState) -> TravelPlanningState:
        """Generate complete itinerary"""
        try:
            llm = GoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            
            prompt = PromptTemplate.from_template(
                """
You are an expert travel planner.

Destination: {destination}
Day: {day} of {num_days}
Travel Date: {date}
Traveler Interests: {interests}
Budget: {budget}

Top Search Results (for planning inspiration):
{search_results}

Now create a structured travel itinerary for this day.
Include:
- Morning, Afternoon, and Evening activities
- Local experiences or food options
- Keep it realistic, engaging, and budget-conscious.
"""
            )
            
            full_itinerary = ""
            
            for day in range(1, state["num_days"] + 1):
                current_date = state["start_date"] + timedelta(days=day-1)
                inputs = {
                    "destination": state["destination"],
                    "day": day,
                    "num_days": state["num_days"],
                    "date": current_date.strftime("%Y-%m-%d"),
                    "interests": ", ".join(state["interests"]),
                    "budget": state["budget"],
                }

                result = generate_daily_itinerary(inputs, llm, prompt)
                full_itinerary += f"Day {day} ({current_date.strftime('%B %d, %Y')}) Itinerary:\n{result}\n\n"
            
            return {
                **state,
                "full_itinerary": full_itinerary,
                "itinerary_generated": True,
                "current_step": "itinerary_generated"
            }
        except Exception as e:
            return {
                **state,
                "full_itinerary": "",
                "itinerary_generated": False,
                "current_step": "itinerary_error",
                "error": str(e)
            }
    
    # -----------------------------
    # Cost Calculation Node
    # -----------------------------
    def cost_calculation_node(state: TravelPlanningState) -> TravelPlanningState:
        """Calculate trip costs"""
        try:
            llm = GoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            
            cost_calculator = TravelCostCalculator()
            cost_result = cost_calculator.calculate_trip_cost(
                state["full_itinerary"],
                state["destination"],
                state["num_days"],
                state["budget"],
                llm
            )
            
            return {
                **state,
                "cost_result": cost_result,
                "current_step": "cost_calculated"
            }
        except Exception as e:
            return {
                **state,
                "cost_result": None,
                "current_step": "cost_error",
                "error": str(e)
            }
    
    # -----------------------------
    # Decision Nodes
    # -----------------------------
    def should_generate_itinerary(state: TravelPlanningState) -> str:
        """Decide whether to proceed with itinerary generation"""
        if state.get("weather_validated", False):
            return "generate_itinerary"
        else:
            return "weather_failed"
    
    def should_calculate_costs(state: TravelPlanningState) -> str:
        """Decide whether to proceed with cost calculation"""
        if state.get("itinerary_generated", False):
            return "calculate_costs"
        else:
            return "itinerary_failed"
    
    # -----------------------------
    # Add nodes to graph
    # -----------------------------
    workflow.add_node("weather_check", weather_check_node)
    workflow.add_node("generate_itinerary", itinerary_generation_node)
    workflow.add_node("calculate_costs", cost_calculation_node)
    
    # Set entry point
    workflow.set_entry_point("weather_check")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "weather_check",
        should_generate_itinerary,
        {
            "generate_itinerary": "generate_itinerary",
            "weather_failed": END
        }
    )
    
    workflow.add_conditional_edges(
        "generate_itinerary",
        should_calculate_costs,
        {
            "calculate_costs": "calculate_costs",
            "itinerary_failed": END
        }
    )
    
    workflow.add_edge("calculate_costs", END)
    
    return workflow.compile()

# Initialize the graph
travel_planner_graph = create_travel_planner_graph()

# -----------------------------
# 4️⃣ Existing Functions (Keep as is)
# -----------------------------
@traceable(run_type="tool", name="weather_check")
def check_weather_conditions(destination, start_date, num_days):
    """Check weather forecast for the destination and dates."""
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
    
    if not OPENWEATHERMAP_API_KEY:
        return True, "⚠️ Weather check skipped (API key not found)", []
    
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={destination}&limit=1&appid={OPENWEATHERMAP_API_KEY}"
        geo_response = requests.get(geo_url)
        geo_data = geo_response.json()
        
        if not geo_data:
            return True, f"⚠️ Could not find location: {destination}", []
        
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        
        forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
        forecast_response = requests.get(forecast_url)
        forecast_data = forecast_response.json()
        
        unfavorable_conditions = ["Rain", "Snow", "Thunderstorm", "Drizzle"]
        daily_weather = {}
        
        for item in forecast_data['list']:
            date = datetime.fromtimestamp(item['dt']).date()
            days_from_start = (date - start_date).days
            if 0 <= days_from_start < num_days:
                weather_main = item['weather'][0]['main']
                if date not in daily_weather:
                    daily_weather[date] = []
                daily_weather[date].append(weather_main)
        
        unfavorable_days = 0
        weather_details = []
        
        for date, conditions in sorted(daily_weather.items()):
            unfavorable_count = sum(1 for c in conditions if c in unfavorable_conditions)
            is_unfavorable = unfavorable_count > len(conditions) / 2
            
            if is_unfavorable:
                unfavorable_days += 1
            
            main_condition = max(set(conditions), key=conditions.count)
            weather_details.append({
                'date': date,
                'condition': main_condition,
                'is_unfavorable': is_unfavorable
            })
        
        is_favorable = unfavorable_days < (num_days / 2)
        
        if is_favorable:
            message = f"✅ Weather looks good! {unfavorable_days} out of {num_days} days may have unfavorable conditions."
        else:
            message = f"⚠️ Weather Alert: {unfavorable_days} out of {num_days} days show unfavorable conditions (rain/snow/storms). Consider changing dates or destination."
        
        return is_favorable, message, weather_details
        
    except Exception as e:
        return True, f"⚠️ Weather check failed: {str(e)}", []

@traceable(run_type="tool", name="duckduckgo_search")
def limited_search(query: str) -> str:
    """Perform a concise DuckDuckGo search and summarize top results."""
    search_tool = DuckDuckGoSearchAPIWrapper(max_results=5)
    results = search_tool.run(query)
    lines = results.split("\n")[:5]
    return "\n".join(lines)

@traceable(run_type="chain", name="itinerary_generation")
def generate_daily_itinerary(inputs, llm, prompt):
    """Generate itinerary for a single day with tracing"""
    try:
        query = f"Best places and activities in {inputs['destination']} related to {inputs['interests']}"
        search_results = limited_search(query)
        
        formatted_prompt = prompt.format(
            destination=inputs['destination'],
            day=inputs['day'],
            num_days=inputs['num_days'],
            date=inputs['date'],
            interests=inputs['interests'],
            budget=inputs['budget'],
            search_results=search_results
        )
        
        with tracing_v2_enabled():
            result = llm.invoke(formatted_prompt)
        
        return result
    except Exception as e:
        return f"Error generating itinerary for day {inputs['day']}: {str(e)}"

def find_local_truetype_font(preferred_names=None):
    """Find a Unicode-capable TTF font installed on the local system."""
    if preferred_names is None:
        preferred_names = [
            "DejaVuSans.ttf", "Arial Unicode.ttf", "Arial Unicode MS.ttf",
            "arialuni.ttf", "Arial.ttf", "LiberationSans-Regular.ttf",
            "NotoSans-Regular.ttf", "Segoe UI Emoji.ttf", "SegoeUIEmoji.ttf",
            "Segoe UI Symbol.ttf"
        ]

    system = platform.system().lower()
    candidates = []
    
    if system == "windows":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates += [os.path.join(windir, "Fonts", name) for name in preferred_names]
    elif system == "darwin":
        mac_dirs = ["/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
        for d in mac_dirs:
            candidates += [os.path.join(d, name) for name in preferred_names]
    else:
        unix_dirs = [
            "/usr/share/fonts/truetype", "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
        ]
        for d in unix_dirs:
            if os.path.isdir(d):
                for root, _, files in os.walk(d):
                    for name in preferred_names:
                        path = os.path.join(root, name)
                        candidates.append(path)

    for p in candidates:
        if p and os.path.isfile(p) and (p.lower().endswith(".ttf") or p.lower().endswith(".otf")):
            return p
    return None

def remove_problematic_chars(text):
    """Remove non-latin1 characters to avoid FPDF errors."""
    return text.encode("latin-1", "ignore").decode("latin-1")

@traceable(run_type="tool", name="pdf_generation")
def save_itinerary_as_pdf(itinerary_text, filename="itinerary.pdf"):
    """Save itinerary_text into a PDF using a local TrueType font if available."""
    from fpdf import FPDF
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_path = find_local_truetype_font()
    if font_path:
        try:
            pdf.add_font("UniFont", "", font_path, uni=True)
            pdf.set_font("UniFont", size=12)
            for line in itinerary_text.splitlines():
                cleaned_line = line.replace('\r', '')
                pdf.multi_cell(0, 8, cleaned_line)
            pdf.output(filename)
            return filename
        except Exception as e:
            print(f"Failed to use font {font_path}: {e}")

    safe_text = remove_problematic_chars(itinerary_text)
    pdf.set_font("Arial", size=12)
    for line in safe_text.splitlines():
        pdf.multi_cell(0, 8, line)
    pdf.output(filename)
    return filename

# -----------------------------
# 5️⃣ User Inputs (Main Screen)
# -----------------------------
st.header("📝 Plan Your Trip")

col1, col2 = st.columns(2)
with col1:
    destination = st.text_input("Enter Destination", "Bali, Indonesia")

with col2:
    budget = st.selectbox("Budget Level", ["Low", "Medium", "High"], index=1)

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("Start Date", datetime.now() + timedelta(days=7))
with col4:
    num_days = st.slider("Trip Duration (Days)", 1, 7, 3)

interests = st.multiselect(
    "Select Your Interests",
    ["Beaches", "Culture", "Adventure", "Food", "Relaxation", "Nightlife"],
    default=["Beaches", "Culture"],
)

# Weather check will trigger automatically below
st.divider()

# -----------------------------
# 6️⃣ Auto Weather Validation
# -----------------------------
if destination and start_date and num_days:
    with st.spinner("🌤️ Checking weather conditions..."):
        is_favorable, message, weather_details = check_weather_conditions(
            destination, start_date, num_days
        )

        st.session_state.weather_validated = is_favorable
        st.session_state.weather_message = message

        if is_favorable:
            st.success(message)
        else:
            st.warning(message)

        if weather_details:
            st.subheader("📅 Weather Forecast")
            for day_info in weather_details:
                icon = "⚠️" if day_info['is_unfavorable'] else "✅"
                st.write(f"{icon} **{day_info['date']}**: {day_info['condition']}")

        if not is_favorable:
            st.warning("❌ Please select a different destination or dates for better weather.")
else:
    st.info("🕓 Waiting for destination and dates to check weather...")

# -----------------------------
# 7️⃣ LangGraph Execution
# -----------------------------
generate_btn = st.button("🚀 Generate Complete Travel Plan (LangGraph)", disabled=not st.session_state.weather_validated)

if generate_btn:
    if not st.session_state.weather_validated:
        st.error("⚠️ Please validate weather conditions first!")
    elif not destination or not interests:
        st.warning("Please provide both destination and interests.")
    else:
        with st.spinner("🎯 Executing complete travel planning workflow with LangGraph..."):
            try:
                # Prepare initial state for LangGraph
                initial_state = TravelPlanningState(
                    destination=destination,
                    budget=budget,
                    start_date=start_date,
                    num_days=num_days,
                    interests=interests,
                    weather_validated=False,
                    weather_message="",
                    weather_details=[],
                    full_itinerary="",
                    itinerary_generated=False,
                    cost_result=None,
                    current_step="start",
                    error=""
                )
                
                # Execute the complete workflow
                final_state = travel_planner_graph.invoke(initial_state)
                
                # Store the graph state
                st.session_state.graph_state = final_state
                
                # Update session state with results
                st.session_state.full_itinerary = final_state.get("full_itinerary", "")
                st.session_state.itinerary_generated = final_state.get("itinerary_generated", False)
                
                # Display results
                st.success("✅ Complete travel plan generated successfully!")
                                
                # Display itinerary
                # Display itinerary - Simple bullet point version
                if final_state.get("itinerary_generated"):
                    st.divider()
                    st.header("🗓️ Generated Itinerary")
                    
                    full_itinerary = final_state["full_itinerary"]
                    
                    # Split by "Day X" pattern
                    import re
                    day_pattern = r'(Day \d+.*?)(?=Day \d+|$)'
                    day_matches = re.findall(day_pattern, full_itinerary, re.DOTALL | re.IGNORECASE)
                    
                    if not day_matches:
                        # If regex doesn't work, fall back to simple splitting
                        day_sections = full_itinerary.split('\n\n')
                    else:
                        day_sections = day_matches
                    
                    for i, day_section in enumerate(day_sections):
                        if not day_section.strip():
                            continue
                            
                        # Extract day number and content
                        lines = [line.strip() for line in day_section.split('\n') if line.strip()]
                        
                        if not lines:
                            continue
                            
                        # First line is usually the day header
                        day_header = lines[0]
                        activities = lines[1:] if len(lines) > 1 else []
                        
                        st.subheader(f"📌 {day_header}")
                        
                        if activities:
                            for activity in activities:
                                st.markdown(f"• {activity}")
                        else:
                            st.info("No specific activities planned for this day.")
                        
                        st.markdown("---")

                    # Generate PDF
                    try:
                        pdf_filename = save_itinerary_as_pdf(final_state["full_itinerary"])
                        with open(pdf_filename, "rb") as f:
                            st.download_button(
                                label="📄 Download Itinerary as PDF",
                                data=f,
                                file_name=f"itinerary_{destination.replace(' ', '_')}_{start_date}.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"PDF generation failed: {str(e)}")
                
                # Display cost results
                if final_state.get("cost_result"):
                    st.divider()
                    st.header("💰 Trip Cost Estimation")
                    
                    cost_result = final_state["cost_result"]
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"🏠 Your Currency ({cost_result['user_currency']})")
                        cost_calculator = TravelCostCalculator()
                        st.metric(
                            "Total Estimated Cost",
                            cost_calculator.format_currency(
                                cost_result['total_user_currency'],
                                cost_result['user_currency']
                            )
                        )
                        
                        st.write("**Cost Breakdown:**")
                        breakdown = cost_result['breakdown_user_currency']
                        st.write(f"🏨 Accommodation: {cost_calculator.format_currency(breakdown['accommodation'], cost_result['user_currency'])}")
                        st.write(f"🍽️ Food & Dining: {cost_calculator.format_currency(breakdown['food'], cost_result['user_currency'])}")
                        st.write(f"🎯 Activities: {cost_calculator.format_currency(breakdown['activities'], cost_result['user_currency'])}")
                        st.write(f"🚕 Transportation: {cost_calculator.format_currency(breakdown['transportation'], cost_result['user_currency'])}")
                        st.write(f"🛍️ Miscellaneous: {cost_calculator.format_currency(breakdown['miscellaneous'], cost_result['user_currency'])}")
                    
                    with col2:
                        st.subheader(f"🌍 Destination Currency ({cost_result['dest_currency']})")
                        st.metric(
                            "Total Estimated Cost",
                            cost_calculator.format_currency(
                                cost_result['total_dest_currency'],
                                cost_result['dest_currency']
                            )
                        )
                        
                        st.info(f"💡 This will help you understand local prices at {destination}")
                
                # Show any errors
                if final_state.get("error"):
                    st.error(f"Workflow error: {final_state['error']}")
                    
            except Exception as e:
                st.error(f"❌ LangGraph workflow execution failed: {str(e)}")
            
            # Show workflow steps
            st.subheader("📊 Workflow Execution Steps")
            steps = [
                ("🌤️ Weather Check", final_state.get("weather_validated", False)),
                ("📅 Itinerary Generation", final_state.get("itinerary_generated", False)),
                ("💰 Cost Calculation", final_state.get("cost_result") is not None)
            ]
            
            for step_name, completed in steps:
                status = "✅ Completed" if completed else "❌ Failed"
                st.write(f"{step_name}: {status}")

# -----------------------------
# 8️⃣ LangGraph Visualization
# -----------------------------
if st.session_state.graph_state:
    st.divider()
    st.header("🔄 LangGraph Workflow Visualization")
    
    st.info("""
    **Workflow Structure:**
    1. 🌤️ **Weather Check** → Validates destination weather conditions
    2. 📅 **Itinerary Generation** → Creates daily travel plans  
    3. 💰 **Cost Calculation** → Estimates trip costs in multiple currencies
    
    **Conditional Logic:**
    - If weather check fails → Stop workflow
    - If itinerary generation fails → Stop workflow  
    - If all steps succeed → Complete workflow
    """)
    
    # Show current state
    current_state = st.session_state.graph_state
    st.subheader("📈 Current Workflow State")
    st.json({
        "current_step": current_state.get("current_step", "unknown"),
        "weather_validated": current_state.get("weather_validated", False),
        "itinerary_generated": current_state.get("itinerary_generated", False),
        "cost_calculated": current_state.get("cost_result") is not None
    })

# -----------------------------
# 9️⃣ LangSmith Status Display
# -----------------------------
st.divider()
st.header("📊 LangSmith & LangGraph Tracing")

if os.getenv("LANGCHAIN_API_KEY"):
    st.success("✅ LangSmith tracing with LangGraph is **ENABLED**")
    st.info("""
    **What's being traced in the unified workflow:**
    - 🎯 **Complete LangGraph Workflow** - End-to-end execution
    - 🌤️ Weather API calls and analysis
    - 🔍 DuckDuckGo search queries and results  
    - 🤖 LLM itinerary generation for each day
    - 💰 Cost calculation and currency conversion
    - 📄 PDF generation process
    - 🔄 Conditional decision points
    
    **View in LangSmith:**
    - Individual function traces
    - Complete workflow graph visualization
    - Performance metrics for each node
    - Error tracking across the workflow
    
    Visit your [LangSmith dashboard](https://smith.langchain.com)
    """)
else:
    st.warning("⚠️ LangSmith tracing is **DISABLED**")
    st.info("To enable tracing, set `LANGCHAIN_API_KEY` in your environment variables.")

st.markdown("---")
st.markdown("💡 *Powered by LangChain Runnables + DuckDuckGo + Gemini Flash + OpenWeatherMap + LangSmith Tracing + LangGraph Workflows*")