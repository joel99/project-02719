#%%
# Open MSW05.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('MSW05.csv', delimiter='\t', skipinitialspace=True)
keep_col = [
    'MSW05_Order',
    'MSW05_Family',
    'MSW05_Genus',
    'MSW05_Species',
    '22-1_HomeRange_km2',
    '22-2_HomeRange_Indiv_km2',
]
# Trim leading and trailing whitespace from column names
df.columns = df.columns.str.strip()

print(df.columns)
df = df[keep_col]
invalid_val = -999
selected_trait = '22-2_HomeRange_Indiv_km2'
df = df[df[selected_trait] != invalid_val]

df['MSW05_Genus'] = df['MSW05_Genus'].str.lower().str.strip()
df['MSW05_Species'] = df['MSW05_Species'].str.lower().str.strip()
df['MSW05_Order'] = df['MSW05_Order'].str.lower().str.strip()
df['MSW05_Family'] = df['MSW05_Family'].str.lower().str.strip()


#%%
# Remove outliers
from scipy import stats
z_scores = stats.zscore(df[selected_trait])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
subset_df = df[filtered_entries]
print(df.columns)

# Change to log scale histogram plot colored by family
plt.figure(figsize=(12, 8))
sns.histplot(data=subset_df, x=selected_trait, hue='MSW05_Family', multiple='stack',
             bins=20, log_scale=True, palette='husl')
plt.xlabel(f'{selected_trait} (log scale)')
plt.ylabel('Count')
plt.title(f'Log Scale Histogram of {selected_trait} by Family')
plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#%%
# Change to log scale histogram plot colored by order
plt.figure(figsize=(6, 4))
sns.histplot(data=subset_df, x=selected_trait, hue='MSW05_Order', multiple='stack',
             bins=20, log_scale=True, palette='husl')
plt.xlabel(f'{selected_trait} (log scale)')
plt.ylabel('Count')
plt.title(f'Log Scale Histogram of {selected_trait} by Order')
plt.legend(title='Order', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print unique orders.
print(subset_df['MSW05_Order'].unique())
# Increase font size and improve aesthetics
plt.figure(figsize=(8, 5))  # Increase figure size for better visibility
sns.set(style="whitegrid", font_scale=1.2)  # Set style and increase font scale

sns.histplot(data=subset_df, x=selected_trait, hue='MSW05_Order', multiple='stack',
             bins=20, log_scale=True, palette='deep')

plt.xlabel(f'{selected_trait} (log scale)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title(f'Log Scale Histogram of {selected_trait} by Order', fontsize=16, fontweight='bold')

# Improve legend
# plt.legend(title='Order', title_fontsize='13', fontsize='11', bbox_to_anchor=(1.05, 1), loc='upper left')
# remove legend
plt.legend([], frameon=False)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show plot
plt.show()

# Reset matplotlib style to default after this plot
plt.style.use('default')

#%%
# Create a new DataFrame with family, genus, and order columns
family_genus_order = subset_df[['MSW05_Family', 'MSW05_Genus', 'MSW05_Order']].drop_duplicates()

# Sort the DataFrame by family, genus, and order
family_genus_order_sorted = family_genus_order.sort_values(['MSW05_Family', 'MSW05_Genus', 'MSW05_Order'])

# Display the head of the sorted DataFrame
print("Head of family, genus, and order data:")
print(family_genus_order_sorted.head())

print(f"\nTotal number of unique family-genus-order combinations: {len(family_genus_order_sorted)}")

#%%
# Open Phenotypes_Zoonomia_3.csv and process the species names
zoonomia = pd.read_csv('Phenotypes_Zoonomia_3.csv')

# Split the 'Name' column into 'Genus' and 'Species'
zoonomia[['Genus', 'Species']] = zoonomia['Name'].str.split('_', n=1, expand=True)
# cast lower
zoonomia['Genus'] = zoonomia['Genus'].str.lower()
zoonomia['Species'] = zoonomia['Species'].str.lower()

# Sanitize msw
#%%
# Display the first few rows to verify the split
print("First few rows of Zoonomia data with separated Genus and Species:")
print(zoonomia[['Name', 'Genus', 'Species']].head())

# Get unique genera from Zoonomia
zoonomia_genera = set(zoonomia['Genus'].unique())

# Get unique genera from MSW05 (subset_df)
msw05_genera = set(subset_df['MSW05_Genus'].unique())
# Compare genera
print(f"\nNumber of unique genera in Zoonomia: {len(zoonomia_genera)}")
print(f"Number of unique genera in MSW05: {len(msw05_genera)}")

# Find genera present in both datasets
common_genera = zoonomia_genera.intersection(msw05_genera)
print(f"Number of genera present in both datasets: {len(common_genera)}")




#%%
# Find overlaps in genera x species
# Create sets of genus-species pairs for both datasets
zoonomia_species = set(zip(zoonomia['Genus'], zoonomia['Species']))
msw05_species = set(zip(subset_df['MSW05_Genus'], subset_df['MSW05_Species']))

# Find common species
common_species = zoonomia_species.intersection(msw05_species)

print(f"Number of common species: {len(common_species)}")
print("\nExample of common species:")
print(list(common_species)[:5])

# Calculate percentage overlap
zoonomia_species_count = len(zoonomia_species)
msw05_species_count = len(msw05_species)
overlap_percentage = (len(common_species) / min(zoonomia_species_count, msw05_species_count)) * 100

print(f"\nPercentage overlap: {overlap_percentage:.2f}%")

# Find species unique to each dataset
zoonomia_only_species = zoonomia_species - msw05_species
msw05_only_species = msw05_species - zoonomia_species

print(f"\nNumber of species only in Zoonomia: {len(zoonomia_only_species)}")
print(f"Number of species only in MSW05: {len(msw05_only_species)}")

print("\nExample of species only in Zoonomia:")
print(list(zoonomia_only_species)[:5])

print("\nExample of species only in MSW05:")
print(list(msw05_only_species)[:5])

#%%
# Print an example of a species that is in both datasets in their original df
if common_species:
    example_species = list(common_species)[0]
    genus, species = example_species

    print("\nExample of a species in both datasets:")
    print(f"Genus: {genus}, Species: {species}")

    print("\nZoonomia data:")
    print(zoonomia[(zoonomia['Genus'] == genus) & (zoonomia['Species'] == species)].iloc[0])

    print("\nMSW05 data:")
    print(df[(df['MSW05_Genus'] == genus) & (df['MSW05_Species'] == species)].iloc[0])
else:
    print("No common species found between the datasets.")

#%%
# Merge Zoonomia and MSW05 datasets for common species
zoonomia_msw = pd.merge(zoonomia, df, left_on=['Genus', 'Species'], right_on=['MSW05_Genus', 'MSW05_Species'], how='inner')

# Plot histogram of home range for common species
plt.figure(figsize=(9, 6))
sns.histplot(data=zoonomia_msw, x=selected_trait, hue='MSW05_Order', multiple='stack',
             bins=20, log_scale=True, palette='husl')

# Set plot labels and title
plt.xlabel(f'{selected_trait} (log scale)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title(f'Log Scale Histogram of {selected_trait} by Order (MSW x Zoonomia)', fontsize=16)

# Customize legend
# plt.legend(title='Order', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title_fontsize=14)

# Adjust layout and increase font sizes
plt.tight_layout()
plt.rcParams.update({'font.size': 14})
plt.tick_params(axis='both', which='major', labelsize=12)

