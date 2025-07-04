def suggest_intervention(prob):
    if prob > 0.85:
        return "Call + Transport Support"
    elif prob > 0.70:
        return "Phone Call Reminder"
    elif prob > 0.50:
        return "SMS Reminder"
    else:
        return "No Action"

def apply_interventions(df, prob_col='no_show_prob'):
    df['intervention'] = df[prob_col].apply(suggest_intervention)
    return df
