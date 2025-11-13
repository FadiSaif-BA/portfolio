import pandas as pd
from typing import List, Dict, Any

class AnalysisTools:
    """
    A utility class for performing common, complex data analysis tasks 
    on survey or transactional data, centralizing core parameters like 
    the DataFrame and standard column names.
    """
    
    def __init__(self, dataframe: pd.DataFrame, total_respondents: int, 
                 question_col: str = 'question',
                 subquest_col: str = 'subquestion',
                 respondent_col: str = 'respondent_id',
                 answer_col: str = 'answer'):
        
        # Core data storage
        self.df = dataframe.copy()
        self.total_respondents = total_respondents
        
        # Standard column name attributes (Can be overridden during initialization)
        self.question_col = question_col
        self.subquest_col = subquest_col
        self.respondent_col = respondent_col
        self.answer_col = answer_col


    # --- 1. Utility Method for Column Cleanup ---
    def sanitize_columns(self) -> pd.DataFrame:
        """
        Cleans column names of the internal DataFrame (self.df) by 
        removing special characters, replacing spaces with underscores, 
        and converting to lowercase.
        """
        new_cols = {}
        for col in self.df.columns:
            name = col.strip()
            name = name.replace(' ', '_').replace('.', '_').replace('-', '_').replace('(', '').replace(')', '')
            name = name.replace('__', '_')
            new_cols[col] = name.lower()
        
        # Apply the renaming directly to the internal DataFrame
        self.df = self.df.rename(columns=new_cols)
        return self.df # Return the modified DataFrame for convenience


    # --- 2. Multiple Response (MR) Analysis ---
    def multi_response_analysis(self, q_list: List[str], group_cols: List[str]) -> pd.DataFrame:
        """
        Calculates frequency and percentage of total sample for MR questions, 
        grouped by specified columns.
        """
        # Access self.df and self.question_col
        df_mr = self.df[self.df[self.question_col].isin(q_list)]

        # Define the full list of columns to group by 
        grouping_keys = group_cols + [self.question_col, self.subquest_col]

        # Calculate frequency based on unique respondent IDs (self.respondent_col)
        mr_frequencies = (
            df_mr.groupby(grouping_keys)[self.respondent_col]
            .nunique()
            .rename('Response Count')
            .to_frame()
        )

        # Calculate Percentage relative to the total sample (self.total_respondents)
        # NOTE: The provided percentage logic using .transform(lambda x: x / x.sum()) 
        # is for % within the question group, NOT % of total sample. 
        # For MR analysis, it should be relative to self.total_respondents.
        # However, to maintain the logic structure you provided:
        
        # If the goal is % of selections WITHIN group_cols and question_col:
        # percent_level = len(group_cols) + 1 # Level of the question_col
        # mr_frequencies['Percentage (%)'] = (
        #     mr_frequencies.groupby(level=percent_level)['Response Count']
        #     .transform(lambda x: (x / x.sum()) * 100)
        # ).round(2)
        
        # Standard MR % (using total sample):
        mr_frequencies['Percentage (%)'] = (
            (mr_frequencies['Response Count'] / self.total_respondents) * 100
        ).round(2)
        
        return mr_frequencies.sort_values(by=grouping_keys, ascending=True)


    # --- 3. Nominal Frequency Analysis (Grouped) ---
    def nominal_frequency_analysis(self, q_list: List[str], group_col: str) -> pd.DataFrame:
        """
        Calculates frequency and percentage for single-choice questions. 
        Percentages sum to 100% within each question group.
        """
        df_filtered = self.df[self.df[self.question_col].isin(q_list)].copy()

        raw_counts = df_filtered.groupby([self.question_col, group_col, self.answer_col]).size().rename('Count')
        counts_df = raw_counts.to_frame()

        # Level 0 is the 'question' column
        counts_df['Question_Total'] = counts_df.groupby(level=0)['Count'].transform('sum')

        counts_df['Percentage (%)'] = (
            (counts_df['Count'] / counts_df['Question_Total'] * 100)
        ).round(2)

        return counts_df.drop(columns=['Question_Total']).sort_values(by=self.question_col)


    # --- 4. Nominal Frequency Analysis (Pivot Table Method) ---
    def nominal_frequency_pivot(self, q_list: List[str], group_col: str) -> pd.DataFrame:
        """
        Calculates frequency using pivot_table. Percentages sum to 100% within 
        each question group.
        """
        df_filtered = self.df[self.df[self.question_col].isin(q_list)].copy()

        # Use pivot_table to get the raw counts (Numerator)
        raw_counts_pivot = pd.pivot_table(
            df_filtered,
            index=[self.question_col, group_col, self.answer_col],
            values=self.respondent_col, # Use respondent_col for counting non-nulls
            aggfunc='count'
        ).rename(columns={self.respondent_col: 'Count'})
        
        # Calculate the total for each question group (Denominator)
        raw_counts_pivot['Question_Total'] = raw_counts_pivot.groupby(level=0)['Count'].transform('sum')

        # Calculate the percentage
        raw_counts_pivot['Percentage (%)'] = (
            (raw_counts_pivot['Count'] / raw_counts_pivot['Question_Total'] * 100)
        ).round(2)

        return raw_counts_pivot.drop(columns=['Question_Total']).sort_index(level=0)


    # --- 5. Grouped Summary Statistics ---
    def grouped_summary_stats(self, numeric_col_list: List[str], group_col: str, 
                              summary_funcs: List[str] = ['mean', 'std', 'count']) -> pd.DataFrame:
        """
        Calculates multiple summary statistics for numeric columns, grouped by a categorical column.
        """
        
        # Check if all columns exist
        for col in numeric_col_list + [group_col]:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' not found in the DataFrame.")

        summary_table = (
            self.df.groupby(group_col)[numeric_col_list]
            .agg(summary_funcs)
        )
        
        # Flatten the MultiIndex columns
        summary_table.columns = ['_'.join(col).strip() for col in summary_table.columns.values]
        
        return summary_table.reset_index()