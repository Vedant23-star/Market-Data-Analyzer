from pydantic import BaseModel, Field
from typing import List, Dict

class PerformanceMetrics(BaseModel):
    cagr_strategy: float = Field(..., description="Annualized Return (CAGR) for the strategy.")
    cagr_bnh: float = Field(..., description="Annualized Return (CAGR) for Buy & Hold benchmark.")
    mdd_strategy: float = Field(..., description="Maximum Drawdown (MDD) for the strategy.")
    mdd_bnh: float = Field(..., description="Maximum Drawdown (MDD) for Buy & Hold benchmark.")
    sharpe_strategy: float = Field(..., description="Sharpe Ratio for the strategy.")
    sharpe_bnh: float = Field(..., description="Sharpe Ratio for Buy & Hold benchmark.")
    total_signals: int = Field(..., description="Total trading signals generated.")

class BacktestResult(BaseModel):
    metrics: PerformanceMetrics
    plot_status: str = Field(..., description="Status of image generation.")