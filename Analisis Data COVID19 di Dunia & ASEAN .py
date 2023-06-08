import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

covid_url = "https://storage.googleapis.com/dqlab-dataset/covid19_worldwide_2020.json"
df_covid_worldwide = pd.read_json(covid_url)

#Kasus Covid-19 di Dunia
print("\nKasus Covid-19 di Dunia")

#Membaca Data COVID-19 Dunia
print("Ukuran dataset: %d kolom dan %d baris.\n" % df_covid_worldwide.shape)
print("Lima data teratas:\n", df_covid_worldwide.head())
print("\nLima data terbawah:\n", df_covid_worldwide.tail())

#Reformat Data Frame COVID-19 Dunia
print("\nInformasi data frame awal:")
df_covid_worldwide.info()

df_covid_worldwide = df_covid_worldwide.set_index('date').sort_index()

print("\nInformasi data frame setelah set index kolom date:")
df_covid_worldwide.info()

#Missing Value di DataFrame COVID-19 Dunia
print("\nJumlah missing value tiap kolom:")
print(df_covid_worldwide.isna().sum())

df_covid_worldwide.dropna(inplace=True)

print("\nJumlah missing value tiap kolom setelah didrop:")
print(df_covid_worldwide.isna().sum())

#Membaca Data Countries
countries_url = "https://storage.googleapis.com/dqlab-dataset/country_details.json"
print("\nLima data teratas dari dataframe countries")
df_countries = pd.read_json(countries_url)
print(df_countries.head())

#Merge Covid19 Data dan Countries
print("\nMapping data covid19 dan data country")
df_covid_denormalized = pd.merge(df_covid_worldwide.reset_index(),df_countries, on="geo_id").set_index("date")
print(df_covid_denormalized)

#Menghitung Fatality Ratio
print("\nFatality Ratio")
df_covid_denormalized["fatality_ratio"] =  (df_covid_denormalized["deaths"]/df_covid_denormalized["confirmed_cases"])
print(df_covid_denormalized)

#20 negara Fatality Ratio tertinggi
print("\n20 Negara Fatality Ratio tertinggi")
df_top_20_fatality_rate = df_covid_denormalized.sort_values("fatality_ratio", ascending=False).head(20)
print(df_top_20_fatality_rate[["geo_id", "country_name", "fatality_ratio"]])

#Visualisasi 20 negara Fatality Ratio tertinggi
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
df_top_20_fatality_rate["fatality_ratio"].sort_values().plot(kind="barh", color="coral")
plt.title("\nTop 20 Highest Fatality Rate Countries", fontsize=18, color="b")
plt.xlabel("Fatality Rate", fontsize=14)
plt.ylabel("Country Name", fontsize=14)
plt.grid(axis="x")
plt.tight_layout()
plt.show()

#Kasus Covid-19 di ASEAN
print("\nKasus Covid-19 di ASEAN")

#Selanjutnya adalah membandingkan kasus covid19 di Indonesia (ID) dengan negara-negara tetangga, yaitu:

# MY -> Malaysia,
# SG -> Singapore,
# TH -> Thailand,
# VN -> Vietnam.

asean_country_id = ["ID", "MY", "SG", "TH", "VN"]
filter_list = [(df_covid_denormalized["geo_id"]==country_id).to_numpy() for country_id in asean_country_id]
filter_array = np.column_stack(filter_list).sum(axis=1, dtype="bool")
df_covid_denormalized_asean = df_covid_denormalized[filter_array].sort_index()
print("Cek nilai unik di kolom 'country_name':", df_covid_denormalized_asean["country_name"].unique())
print(df_covid_denormalized_asean.head())

#Kasus pertama di Asean
print("\nThe first case popped up in each of 5 ASEAN countries:")
for country_id in asean_country_id:
    asean_country = df_covid_denormalized_asean[df_covid_denormalized_asean["geo_id"]==country_id]
    first_case = asean_country[asean_country["confirmed_cases"]>0][["confirmed_cases","geo_id","country_name"]]
    print(first_case.head(1))

# Visualisasi Kasus Covid-19 di ASEAN
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,6))
sns.lineplot(data=df_covid_denormalized_asean, 
             x=df_covid_denormalized_asean.index, 
             y="confirmed_cases", 
             hue="country_name",
             linewidth=2)
plt.xlabel('Record Date', fontsize=14)
plt.ylabel('Total Cases', fontsize=14)
plt.title('Comparison of COVID19 Cases in 5 ASEAN Countries', color="b", fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()