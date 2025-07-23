# data_preparation.py
import pandas as pd

def clean_emissions_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Rename columns for consistency
    df = df.rename(columns={
        "Kilotons of Co2": "Total_CO2_kt",
        "Metric Tons Per Capita": "Per_Capita_CO2_t"
    })
    
    # Convert date to datetime and extract year
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    
    # Filter relevant columns
    df = df[['Country', 'Region', 'Year', 'Total_CO2_kt', 'Per_Capita_CO2_t']]
    
    # Convert kilotons to metric tons (1 kiloton = 1000 metric tons)
    df['Total_CO2_t'] = df['Total_CO2_kt'] * 1000
    
    # Calculate per capita emissions in kg
    df['Per_Capita_CO2_kg'] = df['Per_Capita_CO2_t'] * 1000
    
    # Aggregate data by country and year
    country_data = df.groupby(['Country', 'Year']).agg({
        'Total_CO2_t': 'mean',
        'Per_Capita_CO2_kg': 'mean'
    }).reset_index()
    
    # Save cleaned data
    country_data.to_csv("country_emissions.csv", index=False)
    print("Cleaned data saved to country_emissions.csv")
    
    return country_data

if __name__ == "__main__":
    input_file = "Carbon_(CO2)_Emissions_by_Country.csv"
    clean_emissions_data(input_file)