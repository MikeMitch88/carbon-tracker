# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Set page config FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Country CO₂ Emissions Analyzer",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.header {color: #1e3a8a;}
.metric {font-size: 1.5rem !important;}
.tip-box {border-left: 4px solid #10b981; padding: 0.5rem 1rem; margin: 1rem 0;}
.footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;}
</style>
""", unsafe_allow_html=True)

# Rebuild model function (fallback if loading fails)
@st.cache_resource
def rebuild_model(emissions_df):
    """Rebuild the model if loading fails"""
    try:
        # Prepare features
        X = emissions_df[['Country', 'Year']].copy()
        y = emissions_df['Per_Capita_CO2_kg'].copy()
        
        # Create preprocessor with proper transformer objects
        preprocessor = ColumnTransformer(
            transformers=[
                ('country', OneHotEncoder(handle_unknown='ignore'), ['Country']),
                ('year', StandardScaler(), ['Year'])
            ]
        )
        
        # Create pipeline with actual objects
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Fit the model
        model.fit(X, y)
        
        # Save the corrected model
        joblib.dump(model, "country_emissions_model_fixed.pkl")
        
        return model
    except Exception as e:
        st.error(f"Error rebuilding model: {str(e)}")
        return None

# Load data and model
@st.cache_resource
def load_resources():
    try:
        # Load emissions data
        emissions_df = pd.read_csv("country_emissions.csv")
        country_list = sorted(emissions_df['Country'].unique())
        
        # Try to load the model
        try:
            model = joblib.load("country_emissions_model.pkl")
            
            # Test the model with a simple prediction to see if it works
            test_data = pd.DataFrame({
                'Country': ['United States'],
                'Year': [2020]
            })
            
            # This will fail if the model has string transformers
            test_prediction = model.predict(test_data)
            
        except (AttributeError, Exception) as e:
            st.warning("Original model has issues. Rebuilding model...")
            model = rebuild_model(emissions_df)
            
            if model is None:
                st.error("Failed to create a working model")
                return None, emissions_df, country_list
        
        return model, emissions_df, country_list
        
    except FileNotFoundError as e:
        st.error(f"Required files not found: {str(e)}")
        st.info("Please ensure 'country_emissions.csv' and 'country_emissions_model.pkl' are in the same directory")
        return None, None, []
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None, None, []

# Initialize session state
def init_session():
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = []
    if 'user_location' not in st.session_state:
        st.session_state.user_location = "United States"

# Calculate emissions with error handling
def calculate_emissions(model, country, year):
    try:
        # Ensure input format matches training data
        input_data = pd.DataFrame({
            'Country': [country],
            'Year': [year]
        })
        
        # Make prediction
        prediction = model.predict(input_data)
        return prediction[0]
        
    except Exception as e:
        st.error(f"Error calculating emissions: {str(e)}")
        # Return a reasonable fallback value
        return 5000.0  # Global average approximation

# Get country data
def get_country_data(df, country):
    if df is None:
        return pd.DataFrame()
    country_data = df[df['Country'] == country].sort_values('Year')
    return country_data

# Get reduction targets
def get_reduction_targets(current_emissions):
    return {
        "2030 (SDG Target)": current_emissions * 0.5,
        "2040": current_emissions * 0.3,
        "2050 (Net Zero)": current_emissions * 0.1
    }

# Get climate tips
def get_climate_tips(country, current_emissions, avg_emissions):
    tips = []
    
    if current_emissions > avg_emissions * 1.2:
        tips.append(f"🌍 **Reduce energy consumption**: {country}'s emissions are above average. Consider energy-efficient appliances and renewable energy sources.")
    else:
        tips.append(f"🌿 **Maintain sustainable practices**: {country} is performing better than average. Continue efforts to reduce emissions.")
    
    tips.append("🚗 **Promote electric vehicles**: Transportation is a major contributor to emissions. Support EV infrastructure development.")
    tips.append("🌳 **Protect and expand forests**: Natural carbon sinks are crucial for offsetting emissions.")
    tips.append("🏭 **Advocate for industrial regulations**: Support policies that require industries to reduce their carbon footprint.")
    tips.append("💡 **Educate communities**: Raise awareness about sustainable practices and climate action.")
    
    return tips

# Main application
def main():
    # Load resources
    resources = load_resources()
    
    if resources[0] is None:  # If model loading failed
        st.error("Application cannot start without a working model. Please check your data files.")
        return
    
    model, emissions_df, country_list = resources
    
    # Initialize session
    init_session()
    
    # Header
    st.title("🌍 Country CO₂ Emissions Analyzer")
    st.subheader("Explore and Compare National Carbon Emissions for Climate Action (SDG 13)")
    
    # Introduction
    with st.expander("About This Tool & Dataset"):
        st.write("""
        **Dataset**: Carbon (CO₂) Emissions by Country
        - Contains historical CO₂ emissions data for countries worldwide
        - Metrics: Total emissions (kilotons) and per capita emissions (metric tons)
        - Time range: 1990-2019
        
        **How It Works**:
        1. Select a country to analyze its emissions trajectory
        2. Compare with other countries or global averages
        3. Explore reduction targets and climate action strategies
        
        **Supports SDG 13**: Climate Action by providing data-driven insights for policy-making and awareness.
        """)
    
    # Main analysis section
    col1, col2 = st.columns(2)
    with col1:
        selected_country = st.selectbox(
            "Select Country", 
            country_list,
            index=country_list.index("United States") if "United States" in country_list else 0
        )
    with col2:
        analysis_year = st.slider(
            "Analysis Year", 
            min_value=1990, 
            max_value=2030,
            value=2020
        )
    
    # Get country data
    country_data = get_country_data(emissions_df, selected_country)
    
    if not country_data.empty:
        # Calculate current emissions
        if analysis_year in country_data['Year'].values:
            current_emissions = country_data[country_data['Year'] == analysis_year]['Per_Capita_CO2_kg'].values[0]
        else:
            current_emissions = calculate_emissions(model, selected_country, analysis_year)
        
        # Global average
        global_avg = emissions_df['Per_Capita_CO2_kg'].mean()
        
        # Display metrics
        st.header(f"{selected_country}'s Emissions Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{analysis_year} Per Capita Emissions", f"{current_emissions:.1f} kg CO₂")
        col2.metric("Compared to Global Average", 
                   f"{current_emissions/global_avg:.1f}x",
                   "Below average" if current_emissions < global_avg else "Above average")
        col3.metric("Historical Peak", 
                   f"{country_data['Per_Capita_CO2_kg'].max():.1f} kg CO₂",
                   f"in {country_data.loc[country_data['Per_Capita_CO2_kg'].idxmax(), 'Year']}")
    
        # Visualization section
        st.header("📊 Emissions Trend Analysis")
        
        try:
            # Create historical trend chart
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                x='Year', 
                y='Per_Capita_CO2_kg', 
                data=country_data,
                marker='o',
                color='#1f77b4',
                label=selected_country,
                ax=ax
            )
            
            # Add global average
            global_avg_line = emissions_df.groupby('Year')['Per_Capita_CO2_kg'].mean().reset_index()
            sns.lineplot(
                x='Year', 
                y='Per_Capita_CO2_kg', 
                data=global_avg_line,
                color='red',
                linestyle='--',
                label='Global Average',
                ax=ax
            )
            
            # Add current year marker
            if analysis_year > country_data['Year'].max():
                ax.axvline(x=analysis_year, color='green', linestyle='--', alpha=0.7)
                ax.text(analysis_year+0.5, current_emissions, 
                       f'Prediction: {current_emissions:.1f} kg', 
                       verticalalignment='bottom')
            
            ax.set_title(f"Per Capita CO₂ Emissions Trend: {selected_country}")
            ax.set_ylabel("kg CO₂ per capita")
            ax.set_xlabel("Year")
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    
        # Comparison section
        st.header("🌐 Country Comparison")
        compare_countries = st.multiselect(
            "Compare with other countries", 
            country_list,
            default=["China", "India", "Germany"] if all(c in country_list for c in ["China", "India", "Germany"]) else []
        )
        
        if compare_countries:
            try:
                # Prepare comparison data
                comparison_data = []
                for country in [selected_country] + compare_countries:
                    country_emissions = get_country_data(emissions_df, country)
                    if not country_emissions.empty:
                        latest_year = country_emissions['Year'].max()
                        latest_emissions = country_emissions[country_emissions['Year'] == latest_year]['Per_Capita_CO2_kg'].values[0]
                        comparison_data.append({
                            'Country': country,
                            'Latest Year': latest_year,
                            'Per Capita CO₂ (kg)': latest_emissions
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Create comparison chart
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    sns.barplot(
                        x='Country', 
                        y='Per Capita CO₂ (kg)', 
                        data=comparison_df,
                        hue='Country',
                        palette="viridis",
                        legend=False,
                        ax=ax2
                    )
                    ax2.axhline(y=global_avg, color='red', linestyle='--', label='Global Average')
                    ax2.set_title("Per Capita Emissions Comparison (Latest Available Data)")
                    ax2.set_ylabel("kg CO₂ per capita")
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    # Display comparison table
                    st.dataframe(comparison_df.sort_values('Per Capita CO₂ (kg)', ascending=False))
                    
            except Exception as e:
                st.error(f"Error in comparison section: {str(e)}")
        
        # Reduction targets
        st.header("🎯 Emissions Reduction Targets")
        
        try:
            # Get reduction targets
            targets = get_reduction_targets(current_emissions)
            
            # Create target chart
            target_df = pd.DataFrame({
                'Year': list(targets.keys()),
                'Target': list(targets.values())
            })
            
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.lineplot(
                x='Year', 
                y='Target', 
                data=target_df,
                marker='o',
                color='green',
                label='Target Emissions',
                ax=ax3
            )
            ax3.axhline(y=current_emissions, color='blue', linestyle='--', label='Current Emissions')
            ax3.set_title("Recommended Reduction Pathway")
            ax3.set_ylabel("kg CO₂ per capita")
            ax3.legend()
            st.pyplot(fig3)
            
        except Exception as e:
            st.error(f"Error creating reduction targets chart: {str(e)}")
        
        # Climate action tips
        st.header("💡 Climate Action Recommendations")
        tips = get_climate_tips(selected_country, current_emissions, global_avg)
        
        for tip in tips:
            st.markdown(f"""
            <div class="tip-box">
                <p>{tip}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Download functionality
    st.header("📥 Download Data")
    try:
        if st.download_button(
            "Download Full Emissions Dataset",
            data=open("country_emissions.csv", "rb").read(),
            file_name="country_co2_emissions.csv",
            mime="text/csv"
        ):
            st.success("Dataset downloaded successfully!")
    except FileNotFoundError:
        st.warning("CSV file not found for download")
    
    # Footer
    st.markdown("---")
    st.markdown("### About This Project")
    st.write("""
    **Country CO₂ Emissions Analyzer**  
    This tool helps policymakers, researchers, and citizens understand national carbon emissions patterns.
    
    **Key Features**:
    - Historical emissions trend analysis
    - Country comparison visualization
    - Emissions reduction pathway modeling
    - Climate action recommendations
    
    **Data Source**: Carbon (CO₂) Emissions by Country dataset  
    **Supports**: Sustainable Development Goal 13 (Climate Action)
    """)
    
    st.write("Developed for climate awareness and action")

if __name__ == "__main__":
    main()





















































# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Set page config FIRST (must be first Streamlit command)
# st.set_page_config(
#     page_title="Country CO₂ Emissions Analyzer",
#     page_icon="🌍",
#     layout="wide"
# )

# # Custom CSS
# st.markdown("""
# <style>
# .header {color: #1e3a8a;}
# .metric {font-size: 1.5rem !important;}
# .tip-box {border-left: 4px solid #10b981; padding: 0.5rem 1rem; margin: 1rem 0;}
# .footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;}
# </style>
# """, unsafe_allow_html=True)

# # Load data and model
# @st.cache_resource
# def load_resources():
#     model = joblib.load("country_emissions_model.pkl")
#     emissions_df = pd.read_csv("country_emissions.csv")
#     country_list = sorted(emissions_df['Country'].unique())
#     return model, emissions_df, country_list

# # Initialize session state
# def init_session():
#     if 'comparison_data' not in st.session_state:
#         st.session_state.comparison_data = []
#     if 'user_location' not in st.session_state:
#         st.session_state.user_location = "United States"

# # Calculate emissions
# def calculate_emissions(model, country, year):
#     input_data = pd.DataFrame({
#         'Country': [country],
#         'Year': [year]
#     })
#     return model.predict(input_data)[0]

# # Get country data
# def get_country_data(df, country):
#     country_data = df[df['Country'] == country].sort_values('Year')
#     return country_data

# # Get reduction targets
# def get_reduction_targets(current_emissions):
#     return {
#         "2030 (SDG Target)": current_emissions * 0.5,
#         "2040": current_emissions * 0.3,
#         "2050 (Net Zero)": current_emissions * 0.1
#     }

# # Get climate tips
# def get_climate_tips(country, current_emissions, avg_emissions):
#     tips = []
    
#     if current_emissions > avg_emissions * 1.2:
#         tips.append(f"🌍 **Reduce energy consumption**: {country}'s emissions are above average. Consider energy-efficient appliances and renewable energy sources.")
#     else:
#         tips.append(f"🌿 **Maintain sustainable practices**: {country} is performing better than average. Continue efforts to reduce emissions.")
    
#     tips.append("🚗 **Promote electric vehicles**: Transportation is a major contributor to emissions. Support EV infrastructure development.")
#     tips.append("🌳 **Protect and expand forests**: Natural carbon sinks are crucial for offsetting emissions.")
#     tips.append("🏭 **Advocate for industrial regulations**: Support policies that require industries to reduce their carbon footprint.")
#     tips.append("💡 **Educate communities**: Raise awareness about sustainable practices and climate action.")
    
#     return tips

# # Main application
# def main():
#     # Load resources
#     model, emissions_df, country_list = load_resources()
    
#     # Initialize session
#     init_session()
    
#     # Header
#     st.title("🌍 Country CO₂ Emissions Analyzer")
#     st.subheader("Explore and Compare National Carbon Emissions for Climate Action (SDG 13)")
    
#     # Introduction
#     with st.expander("About This Tool & Dataset"):
#         st.write("""
#         **Dataset**: Carbon (CO₂) Emissions by Country
#         - Contains historical CO₂ emissions data for countries worldwide
#         - Metrics: Total emissions (kilotons) and per capita emissions (metric tons)
#         - Time range: 1990-2019
        
#         **How It Works**:
#         1. Select a country to analyze its emissions trajectory
#         2. Compare with other countries or global averages
#         3. Explore reduction targets and climate action strategies
        
#         **Supports SDG 13**: Climate Action by providing data-driven insights for policy-making and awareness.
#         """)
    
#     # Main analysis section
#     col1, col2 = st.columns(2)
#     with col1:
#         selected_country = st.selectbox(
#             "Select Country", 
#             country_list,
#             index=country_list.index("United States")
#         )
#     with col2:
#         analysis_year = st.slider(
#             "Analysis Year", 
#             min_value=1990, 
#             max_value=2030,
#             value=2020
#         )
    
#     # Get country data
#     country_data = get_country_data(emissions_df, selected_country)
    
#     if not country_data.empty:
#         # Calculate current emissions
#         if analysis_year in country_data['Year'].values:
#             current_emissions = country_data[country_data['Year'] == analysis_year]['Per_Capita_CO2_kg'].values[0]
#         else:
#             current_emissions = calculate_emissions(model, selected_country, analysis_year)
        
#         # Global average
#         global_avg = emissions_df['Per_Capita_CO2_kg'].mean()
        
#         # Display metrics
#         st.header(f"{selected_country}'s Emissions Profile")
#         col1, col2, col3 = st.columns(3)
#         col1.metric(f"{analysis_year} Per Capita Emissions", f"{current_emissions:.1f} kg CO₂")
#         col2.metric("Compared to Global Average", 
#                    f"{current_emissions/global_avg:.1f}x",
#                    "Below average" if current_emissions < global_avg else "Above average")
#         col3.metric("Historical Peak", 
#                    f"{country_data['Per_Capita_CO2_kg'].max():.1f} kg CO₂",
#                    f"in {country_data.loc[country_data['Per_Capita_CO2_kg'].idxmax(), 'Year']}")
    
#     # Visualization section
#     if not country_data.empty:
#         st.header("📊 Emissions Trend Analysis")
        
#         # Create historical trend chart
#         fig, ax = plt.subplots(figsize=(10, 4))
#         sns.lineplot(
#             x='Year', 
#             y='Per_Capita_CO2_kg', 
#             data=country_data,
#             marker='o',
#             color='#1f77b4',
#             label=selected_country,
#             ax=ax
#         )
        
#         # Add global average
#         global_avg_line = emissions_df.groupby('Year')['Per_Capita_CO2_kg'].mean().reset_index()
#         sns.lineplot(
#             x='Year', 
#             y='Per_Capita_CO2_kg', 
#             data=global_avg_line,
#             color='red',
#             linestyle='--',
#             label='Global Average',
#             ax=ax
#         )
        
#         # Add current year marker
#         if analysis_year > country_data['Year'].max():
#             ax.axvline(x=analysis_year, color='green', linestyle='--', alpha=0.7)
#             ax.text(analysis_year+0.5, current_emissions, 
#                    f'Prediction: {current_emissions:.1f} kg', 
#                    verticalalignment='bottom')
        
#         ax.set_title(f"Per Capita CO₂ Emissions Trend: {selected_country}")
#         ax.set_ylabel("kg CO₂ per capita")
#         ax.set_xlabel("Year")
#         ax.legend()
#         st.pyplot(fig)
    
#     # Comparison section
#     st.header("🌐 Country Comparison")
#     compare_countries = st.multiselect(
#         "Compare with other countries", 
#         country_list,
#         default=["China", "India", "Germany"]
#     )
    
#     if compare_countries:
#         # Prepare comparison data
#         comparison_data = []
#         for country in [selected_country] + compare_countries:
#             country_emissions = get_country_data(emissions_df, country)
#             if not country_emissions.empty:
#                 latest_year = country_emissions['Year'].max()
#                 latest_emissions = country_emissions[country_emissions['Year'] == latest_year]['Per_Capita_CO2_kg'].values[0]
#                 comparison_data.append({
#                     'Country': country,
#                     'Latest Year': latest_year,
#                     'Per Capita CO₂ (kg)': latest_emissions
#                 })
        
#         if comparison_data:
#             comparison_df = pd.DataFrame(comparison_data)
            
#             # Create comparison chart (FIXED seaborn palette warning)
#             fig2, ax2 = plt.subplots(figsize=(10, 4))
#             sns.barplot(
#                 x='Country', 
#                 y='Per Capita CO₂ (kg)', 
#                 data=comparison_df,
#                 hue='Country',  # Add hue parameter
#                 palette="viridis",
#                 legend=False,    # Disable legend
#                 ax=ax2
#             )
#             ax2.axhline(y=global_avg, color='red', linestyle='--', label='Global Average')
#             ax2.set_title("Per Capita Emissions Comparison (Latest Available Data)")
#             ax2.set_ylabel("kg CO₂ per capita")
#             ax2.legend()
#             st.pyplot(fig2)
            
#             # Display comparison table
#             st.dataframe(comparison_df.sort_values('Per Capita CO₂ (kg)', ascending=False))
    
#     # Reduction targets
#     if not country_data.empty:
#         st.header("🎯 Emissions Reduction Targets")
        
#         # Get reduction targets
#         targets = get_reduction_targets(current_emissions)
        
#         # Create target chart
#         target_df = pd.DataFrame({
#             'Year': list(targets.keys()),
#             'Target': list(targets.values())
#         })
        
#         fig3, ax3 = plt.subplots(figsize=(10, 4))
#         sns.lineplot(
#             x='Year', 
#             y='Target', 
#             data=target_df,
#             marker='o',
#             color='green',
#             label='Target Emissions',
#             ax=ax3
#         )
#         ax3.axhline(y=current_emissions, color='blue', linestyle='--', label='Current Emissions')
#         ax3.set_title("Recommended Reduction Pathway")
#         ax3.set_ylabel("kg CO₂ per capita")
#         ax3.legend()
#         st.pyplot(fig3)
        
#         # Climate action tips
#         st.header("💡 Climate Action Recommendations")
#         tips = get_climate_tips(selected_country, current_emissions, global_avg)
        
#         for tip in tips:
#             st.markdown(f"""
#             <div class="tip-box">
#                 <p>{tip}</p>
#             </div>
#             """, unsafe_allow_html=True)
    
#     # Download functionality (FIXED button nesting)
#     st.header("📥 Download Data")
#     if st.download_button(
#         "Download Full Emissions Dataset",
#         data=open("country_emissions.csv", "rb").read(),
#         file_name="country_co2_emissions.csv",
#         mime="text/csv"
#     ):
#         st.success("Dataset downloaded successfully!")
    
#     # Footer
#     st.markdown("---")
#     st.markdown("### About This Project")
#     st.write("""
#     **Country CO₂ Emissions Analyzer**  
#     This tool helps policymakers, researchers, and citizens understand national carbon emissions patterns.
    
#     **Key Features**:
#     - Historical emissions trend analysis
#     - Country comparison visualization
#     - Emissions reduction pathway modeling
#     - Climate action recommendations
    
#     **Data Source**: Carbon (CO₂) Emissions by Country dataset  
#     **Supports**: Sustainable Development Goal 13 (Climate Action)
#     """)
    
#     st.write("Developed for climate awareness and action")

# if __name__ == "__main__":
#     main()