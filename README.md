# 🍽️ Zomato Data Analysis Dashboard  

An interactive **Streamlit web app** that analyzes Zomato restaurant data to provide insights into pricing, ratings, services, and customer patterns.  
The app uses clustering, association rules, sentiment analysis (optional), and advanced visualizations.  

---


## 📸 Screenshots

### 🔍 Dashboard Overview  
![Dashboard Overview](screenshots/Screenshot-2025-06-28-141132.png)

### 📊 Visualizations and Clusters  
![Visualizations](screenshots/Screenshot-2025-06-28-141154.png)

---
## 🚀 Features  

✅ **Filters:**  
- Restaurant type (Delivery / Dine-in)  
- Clusters (KMeans groups)  
- Price range slider  

✅ **Visualizations:**  
- Bar chart: Top 10 most common restaurant names  
- Pie chart: Delivery vs Dine-in share  
- Heatmap: Average rating by price band and type  
- Bubble chart: Cost vs rating with vote size  
- Box plot: Price distribution by type  
- Correlation heatmap (rating, cost, votes, sentiment if available)  
- Sentiment histogram (optional — if sentiment data exists)  
- Scatter plot: Cost vs rating colored by cluster  

✅ **Advanced Analytics:**  
- Clustering using KMeans  
- Association rules between services  
- Optional sentiment scoring on synthetic review text  

✅ **Built with:**  
- **Streamlit**  
- **Pandas**  
- **Seaborn / Matplotlib**  
- **scikit-learn**  

---

## ⚙️ How to Run  

### 1️⃣ Install dependencies  
```bash
pip install streamlit pandas seaborn matplotlib scikit-learn
```

### 2️⃣ Launch the app  
```bash
streamlit run app.py
```

### 3️⃣ Interact with filters and visuals in your browser  

---

## 🌟 Example Insights

💡 Restaurants offering both online ordering and table booking tend to have higher ratings.  
💡 Mid-range price bands (~₹300–₹600 for two) are associated with better customer ratings.  
💡 Delivery dominates restaurant types — but adding table booking may enhance perception.

---

## 📊 Possible Extensions

✅ Add map-based visualizations if geo-coordinates are available  
✅ Deploy to Streamlit Cloud or other hosting platforms  
✅ Integrate real review text for NLP sentiment analysis  

---

## 📌 Notes

- This app uses **synthetic sentiment data** if no real review text exists.  
- The dashboard design is **modular and easily extendable**.

---

## ✨ Author

**Zomato Data Analysis Dashboard** built for portfolio and learning purposes.  
👉 Feel free to **fork**, **extend**, or **deploy**!

---
