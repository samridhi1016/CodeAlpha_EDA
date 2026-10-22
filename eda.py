import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/supermarket_sales.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nTotal Sales by Product Line:")

product_sales = df.groupby("Product line")["Total"].sum()

print(product_sales)


product_sales.plot(kind="bar")

plt.title("Total Sales by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



city_sales = df.groupby("City")["Total"].sum().sort_values(ascending=False)

print("\nTotal Sales by City:")
print(city_sales)


city_sales.plot(kind="bar")
plt.title("Total Sales by City")
plt.xlabel("City")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



print("\nAverage Customer Rating:")
print(df["Rating"].mean())

print("\nRating Statistics:")
print(df["Rating"].describe())


plt.figure(figsize=(8, 5))
plt.hist(df["Rating"], bins=10, edgecolor="black")
plt.title("Distribution of Customer Ratings")
plt.xlabel("Customer Rating")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()



rating_by_product = df.groupby("Product line")["Rating"].mean().sort_values(
    ascending=False
)

print("\nAverage Rating by Product Line:")
print(rating_by_product)

rating_by_product.plot(kind="bar")
plt.title("Average Customer Rating by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



payment_counts = df["Payment"].value_counts()

print("\nPayment Method Counts:")
print(payment_counts)
print("\nAll payment categories and counts:")
print(df["Payment"].value_counts(dropna=False).to_string())


payment_counts.plot(kind="bar")
plt.title("Number of Transactions by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print(df["Payment"].unique())
print(df["Payment"].value_counts(dropna=False))



df["Date"] = pd.to_datetime(df["Date"])

print("\nDate Range:")
print("First date:", df["Date"].min())
print("Last date:", df["Date"].max())




df["Month"] = df["Date"].dt.month_name()

monthly_sales = df.groupby("Month")["Total"].sum()


month_order = ["January", "February", "March"]
monthly_sales = monthly_sales.reindex(month_order)

print("\nTotal Sales by Month:")
print(monthly_sales)


monthly_sales.plot(kind="bar")
plt.title("Total Sales by Month")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()




customer_sales = df.groupby("Customer type")["Total"].sum()

print("\nTotal Sales by Customer Type:")
print(customer_sales)

customer_sales.plot(kind="bar")
plt.title("Total Sales by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# Average Transaction Value by Customer Type
avg_transaction = df.groupby("Customer type")["Total"].mean()

print("\nAverage Transaction Value by Customer Type:")
print(avg_transaction)