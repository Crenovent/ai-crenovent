import requests
import json
from fpdf import FPDF

def fetch_api_data(url):
    """Fetch data from the API endpoint."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching API data: {e}")
        return None

def save_as_json(data, filename="evidence_pack.json"):
    """Save the API response as a JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ JSON evidence pack saved as {filename}")

def save_as_pdf(data, filename="evidence_pack.pdf"):
    """Convert the API response to PDF format."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Evidence Pack Report", ln=True, align="C")
    pdf.ln(10)

    # Flatten nested JSON for readability
    def format_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                pdf.cell(200, 8, txt="  " * indent + f"{key}:", ln=True)
                format_dict(value, indent + 1)
            else:
                pdf.cell(200, 8, txt="  " * indent + f"{key}: {value}", ln=True)

    format_dict(data)
    pdf.output(filename)
    print(f"✅ PDF evidence pack saved as {filename}")

def main():
    # Example API endpoint (replace with your actual API)
    api_url = "https://jsonplaceholder.typicode.com/todos/1"

    print("Fetching data from API...")
    data = fetch_api_data(api_url)
    if not data:
        return

    print("\nSelect export format:")
    print("1. PDF")
    print("2. JSON")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        save_as_pdf(data)
    elif choice == "2":
        save_as_json(data)
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
