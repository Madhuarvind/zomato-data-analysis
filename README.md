# Retail Sales Analysis

This project performs an analysis of retail sales data.

## Project Structure


## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd retail-sales-analysis
    ```
2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

-   To run the SQL queries, execute:
    ```bash
    python run_all_queries.py
    ```
-   If `app.py` is a web application (e.g., Flask or Dash), you might run it with:
    ```bash
    python app.py
    ```
    (Further instructions would depend on the nature of `app.py`)

## Data

-   The original dataset is located in `dataset/Superstore.csv`.
-   The cleaned dataset used for analysis is in `cleaned dataset/cleaned_superstore.csv`.

## Analysis

SQL queries for the analysis are stored in `sql/retail_analysis.sql`.

## Visualization

An example visualization output can be found in `visualization/retail_sales_analysis.png`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE)