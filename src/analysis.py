def analyze_sales_trends(df):
    """Analyze sales trends"""
    sales_by_type = df.groupby("Type")["Total_amt_of_sale"].sum()
    return sales_by_type


def top_performing_products(df):
    """Get top performing products"""
    return df["top_selling_product"].value_counts().head(10)
