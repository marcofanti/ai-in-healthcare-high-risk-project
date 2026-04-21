# /// script
# dependencies = [
#   "jinja2",
# ]
# ///

import json
import os
from pathlib import Path
from datetime import datetime
from jinja2 import Template

def get_summary_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get modification time
        mtime = os.path.getmtime(json_path)
        date_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        # Look for output_viz.html files in subdirectories
        exp_dir = json_path.parent
        viz_links = []
        for viz_file in exp_dir.glob("**/output_viz.html"):
            # Create a relative link from eval/all-summary.html to the viz file
            rel_link = os.path.relpath(viz_file, "eval")
            viz_links.append({
                "name": viz_file.parent.name,
                "path": rel_link
            })
            
        return {
            "exp_name": data.get("exp_name", exp_dir.name),
            "date": date_str,
            "mtime": mtime,
            "overall_acc": data.get("overall_acc", 0),
            "judge_acc": data.get("judge_acc", 0),
            "total_samples": data.get("total_samples", 0),
            "per_model_acc": data.get("per_model_acc", {}),
            "viz_links": sorted(viz_links, key=lambda x: x['name']),
            "path": str(json_path)
        }
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return None

def generate_dashboard():
    eval_path = Path("eval")
    summary_files = list(eval_path.glob("**/summary.json"))
    
    summaries = []
    for f in summary_files:
        data = get_summary_data(f)
        if data:
            summaries.append(data)
            
    # Sort by modification date (newest first)
    summaries.sort(key=lambda x: x['mtime'], reverse=True)
    
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Evaluation Summaries Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; }
            .card { transition: transform 0.2s; border: none; }
            .card:hover { transform: translateY(-5px); }
            .metric-label { font-size: 0.85rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
            .metric-value { font-size: 1.25rem; font-weight: bold; color: #212529; }
            .model-acc-item { font-size: 0.9rem; padding: 4px 0; border-bottom: 1px solid #eee; }
            .model-acc-item:last-child { border-bottom: none; }
            .viz-link { text-decoration: none; font-size: 0.9rem; }
            .viz-link:hover { text-decoration: underline; }
            .badge-acc { font-size: 1rem; }
        </style>
    </head>
    <body class="py-5">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center mb-5">
                <div>
                    <h1 class="fw-bold">Evaluation Dashboard</h1>
                    <p class="text-muted">Aggregated summaries from all pathology ensemble experiments.</p>
                </div>
                <div class="text-end">
                    <span class="badge bg-dark">Total Experiments: {{ summaries|length }}</span>
                    <br>
                    <small class="text-muted">Last updated: {{ now }}</small>
                </div>
            </div>

            <div class="row">
                {% for s in summaries %}
                <div class="col-12 mb-4">
                    <div class="card shadow-sm">
                        <div class="card-header bg-white py-3 d-flex justify-content-between align-items-center">
                            <h5 class="mb-0 fw-bold text-primary">{{ s.exp_name }}</h5>
                            <span class="text-muted small">{{ s.date }}</span>
                        </div>
                        <div class="card-body">
                            <div class="row text-center mb-4">
                                <div class="col-md-4">
                                    <div class="metric-label">Overall Acc</div>
                                    <div class="metric-value text-success">{{ "%.2f%%"|format(s.overall_acc * 100) }}</div>
                                </div>
                                <div class="col-md-4">
                                    <div class="metric-label">Judge Acc</div>
                                    <div class="metric-value text-info">{{ "%.2f%%"|format(s.judge_acc * 100) }}</div>
                                </div>
                                <div class="col-md-4">
                                    <div class="metric-label">Samples</div>
                                    <div class="metric-value">{{ s.total_samples }}</div>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6 border-end">
                                    <h6 class="fw-bold mb-3">Model Accuracy Breakdown</h6>
                                    {% for model, acc in s.per_model_acc.items() %}
                                    <div class="model_acc_item d-flex justify-content-between">
                                        <span class="text-truncate me-2" title="{{ model }}">{{ model.split('/')[-1] }}</span>
                                        <span class="fw-bold">{{ "%.2f%%"|format(acc * 100) }}</span>
                                    </div>
                                    {% endfor %}
                                </div>
                                <div class="col-md-6 ps-4">
                                    <h6 class="fw-bold mb-3">Interactive Visual Audits</h6>
                                    {% if s.viz_links %}
                                        <div class="list-group list-group-flush">
                                        {% for viz in s.viz_links %}
                                            <a href="{{ viz.path }}" class="list-group-item list-group-item-action viz-link py-2">
                                                🔍 {{ viz.name }}
                                            </a>
                                        {% endfor %}
                                        </div>
                                    {% else %}
                                        <p class="text-muted small">No visual audits found.</p>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """
    
    template = Template(html_template)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    output_html = template.render(summaries=summaries, now=now)
    
    output_path = Path("eval/all-summary.html")
    with open(output_path, 'w') as f:
        f.write(output_html)
    
    print(f"Aggregate dashboard generated: {output_path}")

if __name__ == "__main__":
    generate_dashboard()
