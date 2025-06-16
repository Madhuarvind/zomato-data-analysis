-- SALES ANALYSIS QUERIES

-- Top 10 Cities by Total Sales
SELECT City, ROUND(SUM(Sales), 2) AS Total_Sales
FROM sales_data
GROUP BY City
ORDER BY Total_Sales DESC
LIMIT 10;

-- Monthly Sales Trend (Year-Wise)
SELECT 
  `Order Year`, 
  `Order Month`, 
  ROUND(SUM(Sales), 2) AS Monthly_Sales
FROM sales_data
GROUP BY `Order Year`, `Order Month`
ORDER BY `Order Year`, `Order Month`;

-- Average Discount by Sub-Category
SELECT `Sub-Category`, ROUND(AVG(Discount), 2) AS Avg_Discount
FROM sales_data
GROUP BY `Sub-Category`
ORDER BY Avg_Discount DESC;

-- Category-wise Profit Contribution
SELECT Category, ROUND(SUM(Profit), 2) AS Total_Profit
FROM sales_data
GROUP BY Category
ORDER BY Total_Profit DESC;

-- Sales vs Profit per Segment
SELECT Segment, ROUND(SUM(Sales), 2) AS Total_Sales, ROUND(SUM(Profit), 2) AS Total_Profit
FROM sales_data
GROUP BY Segment;

-- Total Sales by Region
SELECT Region, ROUND(SUM(Sales), 2) AS Total_Sales
FROM sales_data
GROUP BY Region;

-- Profit Margin by Category
SELECT Category, ROUND(AVG(`Profit Margin`), 2) AS Avg_Margin
FROM sales_data
GROUP BY Category;

-- Monthly Sales Trend
SELECT `Order Year`, `Order Month`, SUM(Sales) AS Monthly_Sales
FROM sales_data
GROUP BY `Order Year`, `Order Month`
ORDER BY `Order Year`, `Order Month`;

-- CUSTOMER ANALYSIS

-- Top 5 Customers by Profit
SELECT `Customer Name`, ROUND(SUM(Profit), 2) AS Total_Profit
FROM sales_data
GROUP BY `Customer Name`
ORDER BY Total_Profit DESC
LIMIT 5;

-- Customers with Negative Profit
SELECT `Customer Name`, SUM(Profit) AS Total_Profit
FROM sales_data
GROUP BY `Customer Name`
HAVING SUM(Profit) < 0
ORDER BY Total_Profit;

-- PRODUCT PERFORMANCE

-- Top-Selling Sub-Categories
SELECT `Sub-Category`, ROUND(SUM(Sales), 2) AS Total_Sales
FROM sales_data
GROUP BY `Sub-Category`
ORDER BY Total_Sales DESC;

-- Profit Margin by Category
SELECT Category, ROUND(AVG(`Profit Margin`), 2) AS Avg_Profit_Margin
FROM sales_data
GROUP BY Category;

-- High Quantity but Low Profit Products
SELECT `Product Name`, Quantity, Profit
FROM sales_data
WHERE Profit < 0 AND Quantity > 5
ORDER BY Quantity DESC;

-- CUSTOM METRICS

-- Sales Categorized as Low/Medium/High
SELECT `Sales Category`, COUNT(*) AS Total_Orders, ROUND(SUM(Sales), 2) AS Total_Sales
FROM sales_data
GROUP BY `Sales Category`;

-- Region with Highest Average Profit Margin
SELECT Region, ROUND(AVG(`Profit Margin`), 2) AS Avg_Margin
FROM sales_data
GROUP BY Region
ORDER BY Avg_Margin DESC;
