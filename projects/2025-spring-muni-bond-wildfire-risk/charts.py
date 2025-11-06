import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd

# Color scheme from AGENTS.md
PRIMARY_COLOR = '#32006e'
SECONDARY_COLOR = '#4b2e83'
TERTIARY_1 = '#e8e3d3'
TERTIARY_2 = '#85754d'
TERTIARY_3 = '#ffc700'
ACCENT_1 = '#aadb1e'
ACCENT_2 = '#2ad2c9'
ACCENT_3 = '#e93cac'

# Set style for minimalist, sleek appearance
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

def chart_1_scorecard_factor_weightings():
    """
    Pie chart showing the breakdown of factor weightings in the
    US Municipal Utility Revenue Debt Scorecard Overview
    """
    # Data from Exhibit 1, page 4 of the methodology PDF
    factors = ['Financial Strength', 'System Characteristics', 'Management', 'Legal Provisions']
    weightings = [40, 30, 20, 10]

    # Grey shades for all except System Characteristics (highlighted in ACCENT_2)
    colors = ['#a0a0a0', TERTIARY_2, '#c0c0c0', '#808080']

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        weightings,
        labels=factors,
        autopct='%1.0f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'weight': 'normal'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # Style the percentage text
    for i, autotext in enumerate(autotexts):
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_weight('bold')

    # Style the labels
    for text in texts:
        text.set_fontsize(13)
        text.set_weight('medium')

    # Add title
    ax.set_title(
        'US Municipal Utility Revenue Debt\nScorecard Factor Weightings',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('scorecard_factor_weightings.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'scorecard_factor_weightings.png'")
    plt.close()


def chart_2_system_characteristics_breakdown():
    """
    Pie chart showing the breakdown of System Characteristics sub-factors
    """
    # Sub-factors and their weightings
    factors = [
        'Service Area Wealth\n(Median Family Income)',
        'Asset Condition\n(Remaining Useful Life)',
        'System Size (O&M)'
    ]
    weightings = [12.5, 10, 7.5]

    # Use a cohesive color palette - different shades from the scheme
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, TERTIARY_1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        weightings,
        labels=factors,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'weight': 'normal'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_weight('bold')

    # Style the labels
    for text in texts:
        text.set_fontsize(13)
        text.set_weight('medium')

    # Add title
    ax.set_title(
        'System Characteristics\nSub-Factor Breakdown',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('system_characteristics_breakdown.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'system_characteristics_breakdown.png'")
    plt.close()


def chart_3_amount_outstanding_by_state():
    """
    Pie chart showing the breakdown of Amount Outstanding by State
    """
    # Load bond reference data
    df = pd.read_csv('data/wa_ca_or_bond_reference_data.csv')

    # Group by state and sum amount outstanding (note: column has typo "Amout")
    state_totals = df.groupby('State')['Amout Outstanding'].sum().sort_values(ascending=False)

    # Prepare data for pie chart
    states = state_totals.index.tolist()
    amounts = state_totals.values.tolist()

    # Use color palette from the scheme
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_2]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        amounts,
        labels=states,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 14, 'weight': 'normal'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_weight('bold')

    # Style the labels
    for text in texts:
        text.set_fontsize(13)
        text.set_weight('medium')

    # Add title with total amount in billions
    total_billions = sum(amounts) / 1000
    ax.set_title(
        f'Utility Bonds Amount Outstanding by State\nTotal: ${total_billions:.1f}B',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('amount_outstanding_by_state.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'amount_outstanding_by_state.png'")
    plt.close()


def chart_4_bonds_by_maturity_bucket():
    """
    Bar chart showing the number of bonds by maturity date bucket
    """
    # Load bond reference data
    df = pd.read_csv('data/wa_ca_or_bond_reference_data.csv')

    # Parse maturity dates
    df['Maturity Date Parsed'] = pd.to_datetime(df['Maturity Date'])

    # Calculate years to maturity from today
    today = pd.Timestamp.now()
    df['Years to Maturity'] = (df['Maturity Date Parsed'] - today).dt.days / 365.25

    # Create maturity buckets
    def categorize_maturity(years):
        if pd.isna(years):
            return None
        elif years <= 5:
            return '≤ 5 years'
        elif years <= 10:
            return '5-10 years'
        else:
            return '10-30 years'

    df['Maturity Bucket'] = df['Years to Maturity'].apply(categorize_maturity)

    # Count bonds in each bucket
    bucket_counts = df['Maturity Bucket'].value_counts().reindex(['≤ 5 years', '5-10 years', '10-30 years'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Create bar chart
    bars = ax.bar(
        bucket_counts.index,
        bucket_counts.values,
        color=[PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_2],
        edgecolor='white',
        linewidth=2
    )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=14,
            weight='bold',
            color=PRIMARY_COLOR
        )

    # Styling
    ax.set_xlabel('Maturity Period', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Number of Bonds', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        'Number of Bonds by Maturity Date Bucket',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Style remaining spines
    ax.spines['left'].set_color(PRIMARY_COLOR)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('bonds_by_maturity_bucket.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'bonds_by_maturity_bucket.png'")
    plt.close()


def chart_5_daily_trading_volume():
    """
    Area chart showing daily trading volume over time with moving average
    """
    # Load transaction data
    df = pd.read_csv('data/msrb_transaction.csv.gz', compression='gzip',
                     usecols=['trade_date', 'par_traded'])

    # Parse dates
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Calculate daily volume
    daily_volume = df.groupby('trade_date')['par_traded'].sum().reset_index()
    daily_volume['par_traded_billions'] = daily_volume['par_traded'] / 1e9

    # Calculate 30-day moving average
    daily_volume['ma_30'] = daily_volume['par_traded_billions'].rolling(window=30, center=False).mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

    # Plot area chart for daily volume
    ax.fill_between(daily_volume['trade_date'], 0, daily_volume['par_traded_billions'],
                     alpha=0.3, color=ACCENT_2, label='Daily Volume')

    # Plot moving average line
    ax.plot(daily_volume['trade_date'], daily_volume['ma_30'],
            color=PRIMARY_COLOR, linewidth=2.5, label='30-Day Moving Average')

    # Styling
    ax.set_xlabel('Date', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Trading Volume ($ Billions)', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        'Daily Utility Municipal Bond Trading Volume',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PRIMARY_COLOR)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}B'))

    # Add legend
    ax.legend(loc='upper left', frameon=False, fontsize=11)

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Grid for readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('daily_trading_volume.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'daily_trading_volume.png'")
    plt.close()


def chart_6_trading_concentration():
    """
    Concentration curve showing what % of bonds account for what % of trading volume
    """
    # Load transaction data
    df = pd.read_csv('data/msrb_transaction.csv.gz', compression='gzip',
                     usecols=['cusip', 'par_traded'])

    # Calculate total volume per CUSIP
    volume_per_cusip = df.groupby('cusip')['par_traded'].sum().sort_values(ascending=False)

    # Calculate cumulative percentage
    total_volume = volume_per_cusip.sum()
    cumulative_volume = volume_per_cusip.cumsum()
    cumulative_volume_pct = (cumulative_volume / total_volume * 100).values

    # Calculate cumulative percentage of bonds
    n_cusips = len(volume_per_cusip)
    cumulative_cusip_pct = np.arange(1, n_cusips + 1) / n_cusips * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Plot concentration curve
    ax.plot(cumulative_cusip_pct, cumulative_volume_pct,
            color=PRIMARY_COLOR, linewidth=3, label='Actual Distribution')

    # Plot perfect equality line (45-degree line)
    ax.plot([0, 100], [0, 100], color='#808080', linewidth=2, linestyle='--',
            label='Perfect Equality', alpha=0.6)

    # Add annotations for key percentiles
    # Find where 10% of bonds cross
    idx_10 = int(n_cusips * 0.10)
    vol_at_10 = cumulative_volume_pct[idx_10]
    ax.plot([10, 10], [0, vol_at_10], color=ACCENT_2, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot([0, 10], [vol_at_10, vol_at_10], color=ACCENT_2, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.scatter([10], [vol_at_10], color=ACCENT_2, s=100, zorder=5)
    ax.annotate(f'Top 10% of bonds\n{vol_at_10:.1f}% of volume',
                xy=(10, vol_at_10), xytext=(25, vol_at_10 - 15),
                fontsize=11, color=PRIMARY_COLOR,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=ACCENT_2, alpha=0.8),
                arrowprops=dict(arrowstyle='->', color=ACCENT_2, lw=1.5))

    # Styling
    ax.set_xlabel('Cumulative % of Bonds (Ranked by Volume)', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Cumulative % of Trading Volume', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        'Trading Volume Concentration\nUtility Bond Market Liquidity Distribution',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Set axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PRIMARY_COLOR)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Add legend
    ax.legend(loc='lower right', frameon=False, fontsize=11)

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Grid for readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('trading_concentration.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'trading_concentration.png'")
    plt.close()


def chart_7_bond_liquidity_tiers():
    """
    Bar chart showing distribution of bonds by number of trades (liquidity tiers)
    """
    # Load transaction data
    df = pd.read_csv('data/msrb_transaction.csv.gz', compression='gzip',
                     usecols=['cusip'])

    # Count trades per CUSIP
    trades_per_cusip = df['cusip'].value_counts()

    # Categorize into liquidity tiers
    def liquidity_tier(count):
        if count <= 5:
            return '1-5 trades'
        elif count <= 20:
            return '6-20 trades'
        elif count <= 100:
            return '21-100 trades'
        else:
            return '100+ trades'

    tier_counts = trades_per_cusip.apply(liquidity_tier).value_counts()

    # Order tiers
    tier_order = ['1-5 trades', '6-20 trades', '21-100 trades', '100+ trades']
    tier_counts = tier_counts.reindex(tier_order)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Create bar chart with gradient colors
    colors_gradient = [TERTIARY_3, TERTIARY_2, SECONDARY_COLOR, PRIMARY_COLOR]
    bars = ax.bar(tier_counts.index, tier_counts.values, color=colors_gradient,
                  edgecolor='white', linewidth=2)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=14,
            weight='bold',
            color=PRIMARY_COLOR
        )

    # Styling
    ax.set_xlabel('Trading Frequency', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Number of Bonds', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        'Bond Liquidity Distribution\nNumber of Bonds by Trading Frequency',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PRIMARY_COLOR)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Format y-axis with commas
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Rotate x-axis labels slightly for better readability
    plt.xticks(rotation=15, ha='right')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('bond_liquidity_tiers.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'bond_liquidity_tiers.png'")
    plt.close()


def chart_8_volume_heatmap():
    """
    Calendar heatmap showing trading volume intensity by date
    """
    import seaborn as sns

    # Load transaction data
    df = pd.read_csv('data/msrb_transaction.csv.gz', compression='gzip',
                     usecols=['trade_date', 'par_traded'])

    # Parse dates
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # Calculate daily volume
    daily_volume = df.groupby('trade_date')['par_traded'].sum().reset_index()
    daily_volume['par_traded_billions'] = daily_volume['par_traded'] / 1e9

    # Extract year, month, and day of month
    daily_volume['year'] = daily_volume['trade_date'].dt.year
    daily_volume['month'] = daily_volume['trade_date'].dt.month
    daily_volume['day'] = daily_volume['trade_date'].dt.day

    # Focus on recent years for better visualization (last 5 years)
    recent_years = daily_volume['year'].max()
    start_year = recent_years - 4
    daily_volume_recent = daily_volume[daily_volume['year'] >= start_year].copy()

    # Create a pivot table for the heatmap
    # Using year-month on y-axis and day of month on x-axis
    daily_volume_recent['year_month'] = daily_volume_recent['trade_date'].dt.to_period('M').astype(str)

    pivot_data = daily_volume_recent.pivot_table(
        values='par_traded_billions',
        index='year_month',
        columns='day',
        aggfunc='sum'
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')

    # Create heatmap using a custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#ffffff', TERTIARY_1, ACCENT_2, SECONDARY_COLOR, PRIMARY_COLOR]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=n_bins)

    sns.heatmap(pivot_data, cmap=cmap, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Trading Volume ($B)'},
                ax=ax, fmt='.1f', annot=False)

    # Styling
    ax.set_xlabel('Day of Month', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Year-Month', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        'Trading Volume Intensity Heatmap\nDaily Municipal Bond Trading Volume (Last 5 Years)',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Adjust tick colors and rotation
    ax.tick_params(colors=PRIMARY_COLOR, which='both')
    plt.yticks(rotation=0)

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('volume_heatmap.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'volume_heatmap.png'")
    plt.close()


def chart_9_utility_agi_ranked():
    """
    Horizontal bar chart ranking utilities by average adjusted gross income
    """
    # Load AGI data
    agi_df = pd.read_csv('data/utility_service_area_average_adjusted_gross_income.csv')

    # Get latest year data
    latest_year = agi_df['year'].max()
    latest_agi = agi_df[agi_df['year'] == latest_year].copy()

    # Sort by AGI
    latest_agi = latest_agi.sort_values('average_adjusted_gross_income', ascending=True)

    # Calculate quartiles for reference lines
    q1 = latest_agi['average_adjusted_gross_income'].quantile(0.25)
    q2 = latest_agi['average_adjusted_gross_income'].quantile(0.50)
    q3 = latest_agi['average_adjusted_gross_income'].quantile(0.75)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 18), facecolor='white')

    # Create color gradient based on position
    n_utilities = len(latest_agi)
    colors = []
    for i in range(n_utilities):
        # Gradient from yellow (low) to purple (high)
        ratio = i / n_utilities
        if ratio < 0.25:
            colors.append(TERTIARY_3)  # Yellow for bottom quartile
        elif ratio < 0.5:
            colors.append(TERTIARY_2)  # Gold for second quartile
        elif ratio < 0.75:
            colors.append(SECONDARY_COLOR)  # Secondary purple for third quartile
        else:
            colors.append(PRIMARY_COLOR)  # Primary purple for top quartile

    # Create horizontal bar chart
    bars = ax.barh(range(n_utilities),
                   latest_agi['average_adjusted_gross_income'] / 1000,
                   color=colors, edgecolor='white', linewidth=0.5)

    # Add quartile reference lines
    ax.axvline(q1 / 1000, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Q1: ${q1/1000:.0f}k')
    ax.axvline(q2 / 1000, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: ${q2/1000:.0f}k')
    ax.axvline(q3 / 1000, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Q3: ${q3/1000:.0f}k')

    # Styling
    ax.set_xlabel('Average Adjusted Gross Income ($1000s)', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Utility Service Area (Ranked)', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        f'Service Area Wealth Distribution\nUtilities Ranked by Average AGI ({latest_year})',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Remove y-axis labels (too many utilities to label)
    ax.set_yticks([])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}k'))

    # Add legend
    ax.legend(loc='lower right', frameon=False, fontsize=11)

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Grid for readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')

    # Add text annotation about range
    min_agi = latest_agi['average_adjusted_gross_income'].min()
    max_agi = latest_agi['average_adjusted_gross_income'].max()
    ax.text(0.98, 0.02, f'Range: ${min_agi/1000:.0f}k - ${max_agi/1000:.0f}k\n({max_agi/min_agi:.1f}x difference)',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, color=PRIMARY_COLOR, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('utility_agi_ranked.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'utility_agi_ranked.png'")
    plt.close()


def chart_10_agi_distribution():
    """
    Distribution plot showing the shape of AGI distribution with quartile markers
    """
    import seaborn as sns

    # Load AGI data
    agi_df = pd.read_csv('data/utility_service_area_average_adjusted_gross_income.csv')

    # Get latest year data
    latest_year = agi_df['year'].max()
    latest_agi = agi_df[agi_df['year'] == latest_year].copy()

    # Calculate statistics
    q1 = latest_agi['average_adjusted_gross_income'].quantile(0.25)
    q2 = latest_agi['average_adjusted_gross_income'].quantile(0.50)
    q3 = latest_agi['average_adjusted_gross_income'].quantile(0.75)
    mean_agi = latest_agi['average_adjusted_gross_income'].mean()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

    # Create histogram
    ax.hist(latest_agi['average_adjusted_gross_income'] / 1000, bins=30,
            color=ACCENT_2, alpha=0.6, edgecolor='white', linewidth=1)

    # Add KDE curve
    from scipy import stats
    kde = stats.gaussian_kde(latest_agi['average_adjusted_gross_income'] / 1000)
    x_range = np.linspace(latest_agi['average_adjusted_gross_income'].min() / 1000,
                          latest_agi['average_adjusted_gross_income'].max() / 1000, 200)

    # Scale KDE to match histogram
    ax2 = ax.twinx()
    ax2.plot(x_range, kde(x_range), color=PRIMARY_COLOR, linewidth=3, label='Distribution Shape')
    ax2.set_ylabel('Density', fontsize=12, color=PRIMARY_COLOR)
    ax2.tick_params(colors=PRIMARY_COLOR, which='both')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color(PRIMARY_COLOR)

    # Add quartile lines
    ymax = ax.get_ylim()[1]
    ax.axvline(q1 / 1000, color=TERTIARY_2, linestyle='--', linewidth=2, alpha=0.8, label=f'Q1: ${q1/1000:.0f}k')
    ax.axvline(q2 / 1000, color=SECONDARY_COLOR, linestyle='-', linewidth=2.5, alpha=0.8, label=f'Median: ${q2/1000:.0f}k')
    ax.axvline(q3 / 1000, color=TERTIARY_2, linestyle='--', linewidth=2, alpha=0.8, label=f'Q3: ${q3/1000:.0f}k')

    # Styling
    ax.set_xlabel('Average Adjusted Gross Income ($1000s)', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Number of Utilities', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        f'Service Area Wealth Distribution\nAGI Distribution Across Utility Service Areas ({latest_year})',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Remove top spine
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color(PRIMARY_COLOR)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}k'))

    # Add legend
    ax.legend(loc='upper right', frameon=False, fontsize=11)

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('agi_distribution.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'agi_distribution.png'")
    plt.close()


def chart_11_agi_by_state_boxplot():
    """
    Box plot comparing AGI distributions across CA/WA/OR
    """
    # Load AGI data
    agi_df = pd.read_csv('data/utility_service_area_average_adjusted_gross_income.csv')

    # Load fire metrics to get state information
    fire_df = pd.read_csv('data/utility_service_area_fire_metrics.csv.gz', compression='gzip')

    # Get unique utility-state mapping
    state_map = fire_df[['NAME', 'STATE']].drop_duplicates()

    # Get latest year AGI data
    latest_year = agi_df['year'].max()
    latest_agi = agi_df[agi_df['year'] == latest_year].copy()

    # Join with state information
    latest_agi = latest_agi.merge(state_map, on='NAME', how='left')

    # Filter out utilities without state info
    latest_agi = latest_agi.dropna(subset=['STATE'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Prepare data for box plot
    states = ['CA', 'WA', 'OR']
    data_by_state = [latest_agi[latest_agi['STATE'] == state]['average_adjusted_gross_income'] / 1000
                     for state in states]

    # Create box plot
    bp = ax.boxplot(data_by_state, labels=states, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor=ACCENT_3, markersize=8),
                    medianprops=dict(color=PRIMARY_COLOR, linewidth=2.5),
                    boxprops=dict(facecolor=ACCENT_2, edgecolor=PRIMARY_COLOR, linewidth=1.5),
                    whiskerprops=dict(color=PRIMARY_COLOR, linewidth=1.5),
                    capprops=dict(color=PRIMARY_COLOR, linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor=TERTIARY_3, markersize=6, alpha=0.6))

    # Styling
    ax.set_xlabel('State', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_ylabel('Average Adjusted Gross Income ($1000s)', fontsize=14, weight='bold', color=PRIMARY_COLOR)
    ax.set_title(
        f'Service Area Wealth by State\nAGI Distribution Comparison ({latest_year})',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PRIMARY_COLOR)
    ax.spines['bottom'].set_color(PRIMARY_COLOR)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}k'))

    # Adjust tick colors
    ax.tick_params(colors=PRIMARY_COLOR, which='both')

    # Grid for readability
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')

    # Add sample size annotations
    for i, state in enumerate(states):
        n = len(data_by_state[i])
        median_val = data_by_state[i].median()
        ax.text(i + 1, ax.get_ylim()[0] * 0.95, f'n={n}',
                ha='center', va='top', fontsize=10, color=PRIMARY_COLOR, style='italic')

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('agi_by_state_boxplot.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'agi_by_state_boxplot.png'")
    plt.close()


def chart_12_fire_exposure_map():
    """
    Choropleth map showing cumulative wildfire exposure (burn acres) by utility
    """
    # Load wildfire metrics
    fire_df = pd.read_csv('data/utility_service_area_fire_metrics.csv.gz', compression='gzip')

    # Aggregate burn acres by utility
    fire_agg = fire_df.groupby('NAME').agg({
        'BurnBndAc': 'sum',
        'STATE': 'first'
    }).reset_index()

    # Load geojson
    gdf = gpd.read_file('data/Electric_Retail_Service_Territories.geojson')

    # Filter to CA, WA, OR
    gdf = gdf[gdf['STATE'].isin(['CA', 'WA', 'OR'])].copy()

    # Merge fire data with geometries
    gdf_fire = gdf.merge(fire_agg, on='NAME', how='left', suffixes=('', '_fire'))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')

    # Plot utilities without fire data in light gray
    gdf_fire[gdf_fire['BurnBndAc'].isna()].plot(
        ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5, alpha=0.5
    )

    # Plot utilities with fire data using color gradient
    gdf_fire_plot = gdf_fire[gdf_fire['BurnBndAc'].notna()].copy()

    # Create custom colormap from yellow to red to purple
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = [TERTIARY_3, '#ff6600', '#cc0000', SECONDARY_COLOR, PRIMARY_COLOR]
    cmap = LinearSegmentedColormap.from_list('fire', colors_list, N=256)

    gdf_fire_plot.plot(
        column='BurnBndAc',
        ax=ax,
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'Total Acres Burned', 'orientation': 'horizontal',
                     'shrink': 0.6, 'aspect': 30, 'pad': 0.05}
    )

    # Set extent to focus on CA/WA/OR
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 49)

    # Styling
    ax.set_title(
        'Cumulative Wildfire Exposure by Utility\nTotal Acres Burned per Service Territory',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )
    ax.set_xlabel('Longitude', fontsize=12, color=PRIMARY_COLOR)
    ax.set_ylabel('Latitude', fontsize=12, color=PRIMARY_COLOR)
    ax.tick_params(colors=PRIMARY_COLOR)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('fire_exposure_map.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'fire_exposure_map.png'")
    plt.close()


def chart_13_fire_risk_classification():
    """
    Map showing utilities classified into fire risk tiers (Low/Medium/High/Extreme)
    """
    # Load wildfire metrics
    fire_df = pd.read_csv('data/utility_service_area_fire_metrics.csv.gz', compression='gzip')

    # Aggregate metrics by utility to create composite risk score
    fire_agg = fire_df.groupby('NAME').agg({
        'BurnBndAc': 'sum',
        'fire_severity_mean': 'mean',
        'fwi_max': 'max',
        'STATE': 'first'
    }).reset_index()

    # Normalize metrics to 0-1 scale for composite score
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    fire_agg['burn_norm'] = scaler.fit_transform(fire_agg[['BurnBndAc']])
    fire_agg['severity_norm'] = scaler.fit_transform(fire_agg[['fire_severity_mean']])
    fire_agg['fwi_norm'] = scaler.fit_transform(fire_agg[['fwi_max']])

    # Create composite risk score (weighted average)
    fire_agg['risk_score'] = (
        fire_agg['burn_norm'] * 0.4 +  # Burn acres 40%
        fire_agg['severity_norm'] * 0.3 +  # Severity 30%
        fire_agg['fwi_norm'] * 0.3  # FWI 30%
    )

    # Classify into risk tiers
    def risk_tier(score):
        if score < 0.25:
            return 'Low'
        elif score < 0.5:
            return 'Medium'
        elif score < 0.75:
            return 'High'
        else:
            return 'Extreme'

    fire_agg['risk_tier'] = fire_agg['risk_score'].apply(risk_tier)

    # Load geojson
    gdf = gpd.read_file('data/Electric_Retail_Service_Territories.geojson')

    # Filter to CA, WA, OR
    gdf = gdf[gdf['STATE'].isin(['CA', 'WA', 'OR'])].copy()

    # Merge fire data with geometries
    gdf_fire = gdf.merge(fire_agg[['NAME', 'risk_tier', 'risk_score']], on='NAME', how='left')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')

    # Plot utilities without fire data in light gray
    gdf_fire[gdf_fire['risk_tier'].isna()].plot(
        ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5, alpha=0.5, label='No Fire Data'
    )

    # Define fire colors for each risk tier
    risk_colors = {
        'Low': '#FFE066',  # Light yellow
        'Medium': '#FF9933',  # Orange
        'High': '#FF3300',  # Red-orange
        'Extreme': '#990000'  # Dark red
    }

    # Plot each risk tier
    for tier in ['Low', 'Medium', 'High', 'Extreme']:
        tier_data = gdf_fire[gdf_fire['risk_tier'] == tier]
        if len(tier_data) > 0:
            tier_data.plot(
                ax=ax,
                color=risk_colors[tier],
                edgecolor='white',
                linewidth=0.5,
                label=f'{tier} Risk'
            )

    # Set extent to focus on CA/WA/OR
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 49)

    # Styling
    ax.set_title(
        'Wildfire Risk Classification by Utility\nComposite Risk Score (Burn Acres, Severity, FWI)',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )
    ax.set_xlabel('Longitude', fontsize=12, color=PRIMARY_COLOR)
    ax.set_ylabel('Latitude', fontsize=12, color=PRIMARY_COLOR)
    ax.tick_params(colors=PRIMARY_COLOR)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add legend
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor=PRIMARY_COLOR,
              fontsize=11, title='Risk Tier')

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('fire_risk_classification.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'fire_risk_classification.png'")
    plt.close()


def chart_14_vulnerability_index_bivariate():
    """
    Bivariate map combining wildfire risk with service area wealth (AGI)
    Shows credit vulnerability: high fire risk + low wealth = highest risk
    """
    # Load wildfire metrics
    fire_df = pd.read_csv('data/utility_service_area_fire_metrics.csv.gz', compression='gzip')

    # Aggregate fire metrics
    fire_agg = fire_df.groupby('NAME').agg({
        'BurnBndAc': 'sum',
        'fire_severity_mean': 'mean',
        'STATE': 'first'
    }).reset_index()

    # Load AGI data
    agi_df = pd.read_csv('data/utility_service_area_average_adjusted_gross_income.csv')
    latest_year = agi_df['year'].max()
    agi_latest = agi_df[agi_df['year'] == latest_year][['NAME', 'average_adjusted_gross_income']].copy()

    # Merge fire and AGI data
    combined = fire_agg.merge(agi_latest, on='NAME', how='inner')

    # Classify into terciles for bivariate scheme
    combined['fire_category'] = pd.qcut(combined['BurnBndAc'], q=3, labels=['Low', 'Medium', 'High'])
    combined['wealth_category'] = pd.qcut(combined['average_adjusted_gross_income'], q=3, labels=['Low', 'Medium', 'High'])

    # Create bivariate category
    combined['bivariate_cat'] = combined['fire_category'].astype(str) + '-' + combined['wealth_category'].astype(str)

    # Define bivariate color scheme (3x3 grid)
    # Rows: Fire risk (Low to High), Columns: Wealth (Low to High)
    # Most vulnerable: High fire + Low wealth (dark red)
    # Least vulnerable: Low fire + High wealth (light yellow)
    bivariate_colors = {
        'Low-Low': '#FFAA66',     # Medium-low risk
        'Low-Medium': '#FFCC66',  # Low-medium risk
        'Low-High': '#FFEE99',    # Lowest risk
        'Medium-Low': '#FF6633',  # Medium-high risk
        'Medium-Medium': '#FF9966', # Medium risk
        'Medium-High': '#FFBB66', # Low-medium risk
        'High-Low': '#CC0000',    # Highest risk (dark red)
        'High-Medium': '#FF3333', # High risk
        'High-High': '#FF6666'    # Medium-high risk
    }

    # Load geojson
    gdf = gpd.read_file('data/Electric_Retail_Service_Territories.geojson')

    # Filter to CA, WA, OR
    gdf = gdf[gdf['STATE'].isin(['CA', 'WA', 'OR'])].copy()

    # Merge combined data with geometries
    gdf_vuln = gdf.merge(combined[['NAME', 'bivariate_cat', 'fire_category', 'wealth_category']],
                         on='NAME', how='left')

    # Create figure with extra space for legend
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='white')

    # Plot utilities without data in light gray
    gdf_vuln[gdf_vuln['bivariate_cat'].isna()].plot(
        ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5, alpha=0.5
    )

    # Plot each bivariate category
    for cat, color in bivariate_colors.items():
        cat_data = gdf_vuln[gdf_vuln['bivariate_cat'] == cat]
        if len(cat_data) > 0:
            cat_data.plot(
                ax=ax,
                color=color,
                edgecolor='white',
                linewidth=0.5
            )

    # Set extent to focus on CA/WA/OR
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 49)

    # Styling
    ax.set_title(
        'Credit Vulnerability Index\nFire Risk vs Service Area Wealth (Bivariate)',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )
    ax.set_xlabel('Longitude', fontsize=12, color=PRIMARY_COLOR)
    ax.set_ylabel('Latitude', fontsize=12, color=PRIMARY_COLOR)
    ax.tick_params(colors=PRIMARY_COLOR)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Create custom bivariate legend (3x3 grid)
    from matplotlib.patches import Rectangle
    legend_x, legend_y = 0.80, 0.15
    legend_size = 0.04

    # Add legend title
    ax.text(legend_x + 0.06, legend_y + 0.15, 'Vulnerability',
            transform=ax.transAxes, fontsize=11, weight='bold',
            ha='center', color=PRIMARY_COLOR)

    # Draw 3x3 legend grid
    fire_levels = ['High', 'Medium', 'Low']
    wealth_levels = ['Low', 'Medium', 'High']

    for i, fire in enumerate(fire_levels):
        for j, wealth in enumerate(wealth_levels):
            cat_key = f'{fire}-{wealth}'
            rect = Rectangle((legend_x + j*legend_size, legend_y + i*legend_size),
                           legend_size, legend_size,
                           facecolor=bivariate_colors[cat_key],
                           edgecolor='white', linewidth=1,
                           transform=ax.transAxes)
            ax.add_patch(rect)

    # Add axis labels for legend
    ax.text(legend_x - 0.01, legend_y + 0.045, 'Fire Risk →',
            transform=ax.transAxes, fontsize=9, ha='right', va='center',
            rotation=90, color=PRIMARY_COLOR)
    ax.text(legend_x + 0.06, legend_y - 0.01, 'Wealth →',
            transform=ax.transAxes, fontsize=9, ha='center', va='top',
            color=PRIMARY_COLOR)

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('vulnerability_index_bivariate.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'vulnerability_index_bivariate.png'")
    plt.close()


def chart_15_fire_metrics_multipanel():
    """
    Multi-panel (2x2) comparison map showing different fire metrics side-by-side
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Load wildfire metrics
    fire_df = pd.read_csv('data/utility_service_area_fire_metrics.csv.gz', compression='gzip')

    # Aggregate different metrics by utility
    fire_agg = fire_df.groupby('NAME').agg({
        'BurnBndAc': 'sum',
        'fire_severity_mean': 'mean',
        'fwi_mean': 'mean',
        'Event_ID': 'count',  # Number of fire events
        'STATE': 'first'
    }).reset_index()
    fire_agg.rename(columns={'Event_ID': 'fire_event_count'}, inplace=True)

    # Load geojson
    gdf = gpd.read_file('data/Electric_Retail_Service_Territories.geojson')

    # Filter to CA, WA, OR
    gdf = gdf[gdf['STATE'].isin(['CA', 'WA', 'OR'])].copy()

    # Merge fire data with geometries
    gdf_fire = gdf.merge(fire_agg, on='NAME', how='left', suffixes=('', '_fire'))

    # Create fire colormap
    fire_cmap = LinearSegmentedColormap.from_list(
        'fire', ['#FFEE99', '#FFB366', '#FF6600', '#CC0000', '#990000'], N=256
    )

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor='white')
    axes = axes.flatten()

    # Define metrics to plot
    metrics = [
        ('BurnBndAc', 'Total Burn Acres', 'Acres'),
        ('fire_severity_mean', 'Mean Fire Severity', 'Severity'),
        ('fwi_mean', 'Mean Fire Weather Index on Days with Fire', 'FWI'),
        ('fire_event_count', 'Number of Fire Events', 'Events')
    ]

    # Plot each metric in a subplot
    for idx, (metric, title, label) in enumerate(metrics):
        ax = axes[idx]

        # Plot utilities without data in gray
        gdf_fire[gdf_fire[metric].isna()].plot(
            ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5, alpha=0.5
        )

        # Plot utilities with data
        gdf_fire_plot = gdf_fire[gdf_fire[metric].notna()].copy()

        if len(gdf_fire_plot) > 0:
            gdf_fire_plot.plot(
                column=metric,
                ax=ax,
                cmap=fire_cmap,
                edgecolor='white',
                linewidth=0.5,
                legend=True,
                legend_kwds={'label': label, 'orientation': 'horizontal',
                           'shrink': 0.7, 'aspect': 20, 'pad': 0.05}
            )

        # Set extent
        ax.set_xlim(-125, -114)
        ax.set_ylim(32, 49)

        # Styling
        ax.set_title(title, fontsize=14, weight='bold', pad=10, color=PRIMARY_COLOR)
        ax.set_xlabel('Longitude', fontsize=10, color=PRIMARY_COLOR)
        ax.set_ylabel('Latitude', fontsize=10, color=PRIMARY_COLOR)
        ax.tick_params(colors=PRIMARY_COLOR, labelsize=8)

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add grid
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Add overall title
    fig.suptitle(
        'Wildfire Metrics Comparison by Utility',
        fontsize=20,
        weight='bold',
        y=0.98,
        color=PRIMARY_COLOR
    )

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('fire_metrics_multipanel.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'fire_metrics_multipanel.png'")
    plt.close()


def chart_16_agi_geospatial_map():
    """
    Choropleth map showing service area wealth (Average Adjusted Gross Income) by utility
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Load AGI data
    agi_df = pd.read_csv('data/utility_service_area_average_adjusted_gross_income.csv')

    # Get latest year data
    latest_year = agi_df['year'].max()
    agi_latest = agi_df[agi_df['year'] == latest_year][['NAME', 'average_adjusted_gross_income']].copy()

    # Load geojson
    gdf = gpd.read_file('data/Electric_Retail_Service_Territories.geojson')

    # Filter to CA, WA, OR
    gdf = gdf[gdf['STATE'].isin(['CA', 'WA', 'OR'])].copy()

    # Merge AGI data with geometries
    gdf_agi = gdf.merge(agi_latest, on='NAME', how='left')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='white')

    # Plot utilities without AGI data in light gray
    gdf_agi[gdf_agi['average_adjusted_gross_income'].isna()].plot(
        ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5, alpha=0.5
    )

    # Plot utilities with AGI data using color gradient
    gdf_agi_plot = gdf_agi[gdf_agi['average_adjusted_gross_income'].notna()].copy()

    # Create custom colormap from project colors (light to dark purple)
    colors_list = [TERTIARY_1, TERTIARY_3, ACCENT_1, ACCENT_2, SECONDARY_COLOR, PRIMARY_COLOR]
    cmap = LinearSegmentedColormap.from_list('wealth', colors_list, N=256)

    gdf_agi_plot.plot(
        column='average_adjusted_gross_income',
        ax=ax,
        cmap=cmap,
        edgecolor='white',
        linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'Average Adjusted Gross Income ($)', 'orientation': 'horizontal',
                     'shrink': 0.6, 'aspect': 30, 'pad': 0.05,
                     'format': '${x:,.0f}'}
    )

    # Set extent to focus on CA/WA/OR
    ax.set_xlim(-125, -114)
    ax.set_ylim(32, 49)

    # Styling
    ax.set_title(
        f'Service Area Wealth Distribution\nAverage Adjusted Gross Income by Utility ({latest_year})',
        fontsize=18,
        weight='bold',
        pad=20,
        color=PRIMARY_COLOR
    )
    ax.set_xlabel('Longitude', fontsize=12, color=PRIMARY_COLOR)
    ax.set_ylabel('Latitude', fontsize=12, color=PRIMARY_COLOR)
    ax.tick_params(colors=PRIMARY_COLOR)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Add statistics annotation
    min_agi = gdf_agi_plot['average_adjusted_gross_income'].min()
    max_agi = gdf_agi_plot['average_adjusted_gross_income'].max()
    median_agi = gdf_agi_plot['average_adjusted_gross_income'].median()

    # Save with transparent background
    plt.tight_layout()
    plt.savefig('agi_geospatial_map.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Chart saved as 'agi_geospatial_map.png'")
    plt.close()


if __name__ == "__main__":
    #chart_1_scorecard_factor_weightings()
    #chart_2_system_characteristics_breakdown()
    #chart_3_amount_outstanding_by_state()
    #chart_4_bonds_by_maturity_bucket()
    #chart_5_daily_trading_volume()
    #chart_6_trading_concentration()
    #chart_7_bond_liquidity_tiers()
    #chart_8_volume_heatmap()
    #chart_9_utility_agi_ranked()
    #chart_10_agi_distribution()
    #chart_11_agi_by_state_boxplot()
    #chart_12_fire_exposure_map()
    #chart_13_fire_risk_classification()
    #chart_14_vulnerability_index_bivariate()
    chart_15_fire_metrics_multipanel()
    chart_16_agi_geospatial_map()
