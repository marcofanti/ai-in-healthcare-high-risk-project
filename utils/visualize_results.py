# /// script
# dependencies = [
#   "pandas",
#   "jinja2",
# ]
# ///

import pandas as pd
import argparse
import os
import sys

def generate_html(csv_path, show_images=False):
    df = pd.read_csv(csv_path)
    
    # Identify models dynamically
    # Look for columns ending in _Correct
    correct_cols = [c for c in df.columns if c.endswith('_Correct')]
    models = [c.replace('_Correct', '') for c in correct_cols]
    
    def get_status(row):
        judge_correct = (row['Judge Answer'] == row['Right Answer'])
        # Check all models
        model_results = [row[f'{m}_Correct'] == 'T' for m in models]
        all_models_correct = all(model_results)
        all_models_wrong = not any(model_results)
        any_model_correct = any(model_results)
        
        if judge_correct:
            if all_models_correct:
                return "CONSENSUS", "bg-success"
            elif all_models_wrong:
                return "SAVED", "bg-warning text-dark"
            else:
                return "RECONCILED", "bg-info text-white"
        else:
            if any_model_correct:
                return "MISSED", "bg-danger"
            else:
                return "FAILED", "bg-secondary"

    # Add Status and Style classes
    df['Status_Text'], df['Status_Class'] = zip(*df.apply(get_status, axis=1))
    
    # Prepare HTML Template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pathology Ensemble Audit</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css">
        <style>
            .sparkline-container { width: 100px; height: 15px; background: #eee; border-radius: 3px; position: relative; }
            .sparkline-bar { height: 100%; background: #4CAF50; border-radius: 3px; }
            .prob-val { font-size: 0.75rem; color: #666; }
            .correct-cell { background-color: #d1e7dd !important; }
            .wrong-cell { background-color: #f8d7da !important; }
            .path-img { max-width: 150px; cursor: zoom-in; border-radius: 4px; }
            table.dataTable td { vertical-align: middle; }
        </style>
    </head>
    <body class="p-4 bg-light">
        <div class="container-fluid">
            <h2 class="mb-4">Pathology Ensemble Results Audit</h2>
            <div class="card shadow">
                <div class="card-body">
                    <table id="resultsTable" class="table table-hover">
                        <thead class="table-dark">
                            <tr>
                                <th>Status</th>
                                {% if show_images %}<th>Image</th>{% endif %}
                                <th>Question</th>
                                <th>Correct</th>
                                {% for model in models %}
                                    <th>{{ model|upper }}</th>
                                    <th>{{ model|upper }} Prob</th>
                                {% endfor %}
                                <th>Judge</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for idx, row in df.iterrows() %}
                            <tr>
                                <td><span class="badge {{ row['Status_Class'] }}">{{ row['Status_Text'] }}</span></td>
                                {% if show_images %}
                                    <td><img src="{{ row['img_path'] }}" class="path-img" onerror="this.src='https://via.placeholder.com/150?text=No+Image'"></td>
                                {% endif %}
                                <td class="small">{{ row['Question'] }}</td>
                                <td class="text-center fw-bold">{{ row['Right Answer'] }}</td>
                                
                                {% for model in models %}
                                    <td class="{{ 'correct-cell' if row[model+'_Correct'] == 'T' else 'wrong-cell' }} text-center">
                                        {{ row[model+'_Correct'] }}
                                    </td>
                                    <td>
                                        {% for choice in ['A', 'B', 'C', 'D'] %}
                                            <div class="d-flex align-items-center mb-1">
                                                <span class="me-1 fw-bold small">{{ choice }}:</span>
                                                <div class="sparkline-container flex-grow-1">
                                                    <div class="sparkline-bar" style="width: {{ row[model+'_Prob_'+choice] * 100 }}%"></div>
                                                </div>
                                                <span class="ms-1 prob-val">{{ (row[model+'_Prob_'+choice]|float)|round(2) }}</span>
                                            </div>
                                        {% endfor %}
                                    </td>
                                {% endfor %}
                                
                                <td class="{{ 'correct-cell' if row['Judge Answer'] == row['Right Answer'] else 'wrong-cell' }} text-center fw-bold">
                                    {{ row['Judge Answer'] }}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
        <script>
            $(document).ready(function() {
                $('#resultsTable').DataTable({
                    pageLength: 25,
                    order: [[0, 'asc']]
                });
            });
        </script>
    </body>
    </html>
    """
    
    from jinja2 import Template
    template = Template(html_template)
    output_html = template.render(df=df, models=models, show_images=show_images)
    
    out_path = csv_path.replace('.csv', '_viz.html')
    with open(out_path, 'w') as f:
        f.write(output_html)
    
    print(f"Visualization generated: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--images", action="store_true", help="Show pathology images in the table")
    args = parser.parse_args()
    
    if os.path.exists(args.csv_path):
        generate_html(args.csv_path, show_images=args.images)
    else:
        print(f"Error: File {args.csv_path} not found")
