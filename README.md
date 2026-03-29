# 🚚 Delivery Optimization System

A Python + Streamlit application that optimally assigns deliveries to 3 agents using a greedy load-balancing algorithm, with full CSV input/output and interactive visualizations.

---

## 📌 Algorithm & Approach

### Sorting
Deliveries are first sorted by **priority** (High → Medium → Low), then by **distance** (ascending). This ensures urgent packages are evaluated first.

### Greedy Assignment
```
For each delivery (in sorted order):
    Find the agent with the lowest cumulative distance
    Assign this delivery to that agent
    Update that agent's total distance
```
This is an **O(n)** greedy algorithm that approximates the minimum-maximum distance (minimax) problem.

### Optional: Priority-Weighted Balancing
When enabled, distances are multiplied by a weight factor before comparing agents:
- High = 1.0× (favored, assigned early)
- Medium = 1.5×
- Low = 2.0×

This biases High-priority deliveries toward less-loaded agents, improving responsiveness for urgent packages.

---

## 📋 Assumptions

| Assumption | Detail |
|---|---|
| Warehouse location | Default: Bangalore (12.9716°N, 77.5946°E) — configurable in sidebar |
| Distance | Computed via Haversine formula if Lat/Lon present; otherwise uses `Distance from warehouse` column; generates synthetic distances (5–150 km) if neither exists |
| Priority | Uses `Delivery Priority` column if present; auto-generates (30% High, 40% Medium, 30% Low) if missing |
| Agents | Fixed at 3 agents |
| Duplicates | Duplicate Location IDs are dropped (first occurrence kept) |
| Missing values | Filled with column median for distances; random valid priority for priority |

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib
```

### 2. Launch the app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`

### 3. Use the app
1. Upload your CSV **or** check "Use bundled sample CSV"
2. Configure warehouse coordinates and algorithm options in the sidebar
3. View the sorted delivery table
4. Click **⚡ Run Optimization & Assign Deliveries**
5. Explore agent cards, charts, and per-agent breakdowns
6. Download the exported CSV(s)

---

## 📂 CSV Input Format

The script accepts flexible CSV input. Recognized columns:

| Column | Required | Notes |
|---|---|---|
| `Location ID` | ✅ | Unique delivery identifier |
| `Latitude` / `Longitude` | Optional | Used to compute haversine distance |
| `Distance from warehouse` | Optional | Direct distance in km |
| `Delivery Priority` | Optional | `High`, `Medium`, or `Low` |

Any additional columns (Customer Name, Package Weight, etc.) are preserved in the preview.

### Sample Input Row
```
Location ID,Latitude,Longitude,Delivery Priority
LOC001,12.9716,77.5946,High
```

---

## 📤 Sample Output (Delivery Plan CSV)

```
Agent,Location IDs,Total Distance (km),Delivery Count
Agent 1,"LOC007, LOC004, LOC022, ...",312.45,17
Agent 2,"LOC013, LOC016, LOC025, ...",308.90,17
Agent 3,"LOC019, LOC028, LOC031, ...",310.12,16
```

---

## 📁 Project Structure

```
delivery_optimizer/
├── app.py                          ← Main Streamlit application
├── data/
│   └── amazon_deliveries.csv       ← Sample 50-row dataset
├── exports/                        ← Auto-created export folder
└── README.md
```

---

## 🧪 Module Reference

| Function | Description |
|---|---|
| `read_csv(file)` | Reads, validates, and normalizes CSV; computes distances |
| `sort_deliveries(df)` | Sorts by priority then distance |
| `assign_deliveries(df, use_weighted)` | Greedy assignment to 3 agents |
| `export_plan(assignment)` | Converts assignment dict → clean DataFrame |
| `visualize_results(assignment)` | Returns a matplotlib Figure with 2 bar charts |
| `haversine(lat1, lon1, lat2, lon2)` | Computes km distance between two coordinates |

---

## 📦 Dependencies

```
streamlit>=1.32
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
```

---

## 💡 Bonus Optimizations

- **Priority weighting** available via sidebar toggle
- **Load imbalance metric** shown in footer (max − min agent distance)
- **Per-agent breakdown tab** with location + distance + priority detail
- Handles empty CSV, all-missing distances, duplicate IDs, and malformed priorities
