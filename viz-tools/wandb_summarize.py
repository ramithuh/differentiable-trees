import wandb
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from typing import Dict

import plotly.graph_objs as go
import plotly.figure_factory as ff

api = wandb.Api()

def print_stats(counts, seeds):
    """
        Prints the seeds and counts of runs for each leaf complexity.
        Helps to pinpoint if there are any missing runs.
    """
    key_counts = np.array(list(counts.values()))
    
    ## check if all counts are the same
    if(len(np.unique(key_counts))==1):
        st.success(f"All leaves have {key_counts[0]} runs")
    else:
        st.warning("Not all leaves have the same number of runs")
    # else:
    with st.expander("Number of runs per leaf"):
        for key in counts.keys():
            st.write(f"Number of runs for {key} leaves: {counts[key]}")
            
    ## check if there are representative seeds (42 to 51)
    seeds_represented = True
    with st.expander(f"Seed representation for {len(counts.keys())} leaves"):
        for i in range(42, 52):
            st.write(f"Count from seed {i} = {seeds[i]}")
            
            if(i not in seeds.keys()):
                seeds_represented = False
            else:
                ## check if the runs from each seed matches the number of types of experiments (leaf complexity)
                if(seeds[i] != len(counts.keys())):
                    seeds_represented = False
    
    if(seeds_represented):
        st.success("All seeds are represented")
    else:
        st.error("Not all seeds are represented")
                  
def has_run_crashed(run) -> bool:
    if('best cost' not in run.summaryMetrics.keys()):
        return True
    
    return False    

def has_keywords(run_name, keywords) -> bool:
    for keyword in keywords:
        if keyword in run_name:
            return True
    
    return False
    
def retreive_exps(
        runs : list,
        leaves_list : list,
        incl_keywords : list = None,
        excl_keywords : list = None,
    ) -> Dict:
    
    errors = {}
    sankoffs = {}
    elves = {}
    
    counts = {}
    seeds = {}

    for l in sorted(leaves_list):
        errors[str(l)] = []
        sankoffs[str(l)] = []
        elves[str(l)] = []
        counts[str(l)] = 0
        
        for run in runs:            
            if(run.config['n_leaves'] == l and run.config['tLs']==[0, 0.005, 10, 50]):
                
                if(has_keywords(run.name, excl_keywords) or has_run_crashed(run) ):
                    continue
                    
                if(incl_keywords is not None):
                    if(not has_keywords(run.name, incl_keywords)):
                        continue
        
                cost = run.summaryMetrics['best cost']
                gt   = run.summaryMetrics['sankoff_cost']
                
                errors[str(l)].append(cost - gt)
                sankoffs[str(l)].append(gt)
                elves[str(l)].append(cost)
                counts[str(l)] += 1
                
                if(run.config['seed'] not in seeds.keys()):
                    seeds[run.config['seed']] = 1
                else:
                    seeds[run.config['seed']] += 1
            
    return errors, sankoffs, elves, counts, seeds

with st.sidebar:
    ### Sidebar options to quert wandb
    
    # Intro
    st.title('Query Elves Experiments')
    image = Image.open('logo.png')
    image = image.resize((150, 150))
    st.image(image)
    
    # Query options
    project = st.selectbox("Wandb project", ("ramithx1/Batch-Run-Maximum-Parsimony-A100-META-optimized", "N/A"), disabled=True)
    Exclusive = st.radio('Exclusive tree types', ('All', 'Perfect Trees Only', 'Diverse Trees Only'), index = 0)

    
    choices = [4,6,8,12,16,24,32,48,64,96,128,192]
    incl_keywords_default = ["Final", "Diverse-trees-regular"]
    
    if(Exclusive == 'Perfect Trees Only'):
        choices = [4,8,16,32,64,128]
        incl_keywords_default = ["Final"]
    elif(Exclusive == 'Diverse Trees Only'):
        choices = [6,12,24,48,96,192]
        incl_keywords_default = ["Diverse-trees-regular"]
        
    incl_keywords = st.multiselect('Include Keywords (if at least one is there)', ["Final", "Diverse-trees-regular", "S1", "S1-Diverse"] , incl_keywords_default)
    excl_keywords = st.multiselect('Exclude Keywords (if at least one is there)', ["custom", "test"], ["custom", "test"])
    
        
    leaves = st.multiselect('Leaves (N)', choices, choices)
    

runs = api.runs(project)
errors, sankoffs, elves, counts, seeds = retreive_exps(runs, leaves, incl_keywords, excl_keywords)

errors_percentages = {num_leaves: [abs(b - a)/a*100 for a, b in zip(sankoffs[num_leaves], elves[num_leaves])]
          for num_leaves in sankoffs}
    
print_stats(counts, seeds)

df = pd.DataFrame(columns=['Leaves', 'Mean Optimal Solution', 'Our Mean Solution', 'Mean Error', 'Mean Error %'])

for idx, key in enumerate(leaves):
    err  = np.array(errors[str(key)])
    s_gt = np.array(sankoffs[str(key)])
    sol  = np.array(elves[str(key)])
    
    assert err.shape == sol.shape
    assert s_gt.shape == sol.shape
    df.loc[idx] = [key, str(round(s_gt.mean(),1)), str(round(sol.mean(),1)), str(round(err.mean(),2)) + " ± " + str(round(err.std(),1)) , str(round(err.mean()/s_gt.mean()*100,2)) ]
    
st.table(df)


# Define the font and color styles
font_family = "sans-serif"
title_font_size = 20
axis_font_size = 16

## 12 colors
colors = ['#1f77b4', '#ff7f0e','#2ca02c','#d62728', '#9467bd', '#8c564b','#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff0000', '#00ff00']


if(Exclusive == 'All'):
    colors = ['#1f77b4', '#000000', '#ff7f0e', '#000000', '#2ca02c', '#000000', '#d62728', '#000000', '#9467bd', '#000000',  '#8c564b', '#000000', '#e377c2', '#000000', '#7f7f7f', '#bcbd22', '#17becf', '#ff0000', '#00ff00']



# Compute the errors


fig = go.Figure()



# Sort the keys (number of leaves) and loop through them to add a box plot for each group
sorted_keys = sorted(errors.keys(), key=int)

for idx, num_leaves in enumerate(sorted_keys):
    fig.add_trace(go.Box(y=errors_percentages[num_leaves],
                         name=f'{num_leaves} leaves',
                         x=[f'{num_leaves}']*len(errors_percentages[num_leaves]),
                         boxpoints='all', jitter=0.3, pointpos=-1.8,
                         yaxis='y1',
                         showlegend=False,
                         marker_color=colors[idx],
                         marker_size=7,
                         line_width=2,
                         boxmean=True))

fig.update_layout(
    plot_bgcolor='rgba(230, 236, 245, 1)',  # This sets the background color of the plot area
    paper_bgcolor='rgba(255, 255, 255, 1)',  # This sets the background color of the entire figure
)

fig.update_layout(xaxis=dict(title='Number of Leaves',
                             title_font=dict(family=font_family, size=title_font_size),
                             tickfont=dict(family=font_family, size=16),
                             gridcolor='rgb(255,255,255)',
                             
                             showgrid=True,  # Ensure vertical gridlines are shown
                             tickvals=sorted_keys,  # This will ensure only specific ticks are shown
                             ticktext=[str(k) for k in sorted_keys],  # This will show the exact tick labels you want

                             gridwidth=2,
                             showline=True,
                             linewidth=2, 
                             linecolor='white'
                             )
                  )

fig.update_layout(yaxis=dict(title='Mean Δ (%)', side='left', title_standoff=25,
                             title_font=dict(family=font_family, size=title_font_size),
                             tickfont=dict(family=font_family, size=axis_font_size),
                             tickformat='.1f',
                             gridcolor='rgb(255,255,255)',
                             gridwidth=2,
                             showline=True,
                             linewidth=2, 
                             linecolor='white',
                             zeroline=True,  # Show the zero line for y-axis
                             zerolinewidth=2,  # Set the zero line thickness for y-axis
                             zerolinecolor='rgb(255,255,255)'  # Set the zero line color for y-axis
                             ))


st.plotly_chart(fig, use_container_width=True)
