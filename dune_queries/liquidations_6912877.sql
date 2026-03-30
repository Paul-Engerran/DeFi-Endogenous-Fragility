WITH borrow_per_tx AS (
  SELECT
    tx_hash, blockchain, block_time, project, version,
    SUM(ABS(amount_usd)) AS debt_amount_usd,
    COUNT(*)             AS n_debt_events
  FROM lending.borrow
  WHERE transaction_type = 'borrow_liquidation'
    AND block_time >= TIMESTAMP '2021-03-15'
    AND block_time <  TIMESTAMP '2025-12-01'
    AND ABS(amount_usd) > 5
    AND NOT (project = 'uwulend' AND block_time >= TIMESTAMP '2024-06-10')
    AND NOT (project = 'euler'
             AND block_time >= TIMESTAMP '2023-03-13'
             AND block_time <  TIMESTAMP '2023-04-13')
  GROUP BY 1, 2, 3, 4, 5
),
supply_per_tx AS (
  SELECT
    tx_hash, blockchain,
    SUM(ABS(amount_usd)) AS collateral_amount_usd,
    COUNT(*)             AS n_supply_events
  FROM lending.supply
  WHERE transaction_type IN ('deposit_liquidation', 'supply_liquidation')
    AND block_time >= TIMESTAMP '2021-03-15'
    AND block_time <  TIMESTAMP '2025-12-01'
    AND symbol IN ('WETH','ETH','stETH','wstETH','rETH','cbETH','sfrxETH','ETHx')
  GROUP BY 1, 2
)
SELECT
  DATE_TRUNC('hour', b.block_time) AS date,
  SUM(b.debt_amount_usd)           AS total_debt_repaid_usd,
  SUM(s.collateral_amount_usd)     AS total_collateral_seized_usd,
  SUM(b.n_debt_events)             AS n_liquidations
FROM borrow_per_tx b
INNER JOIN supply_per_tx s
  ON  b.tx_hash    = s.tx_hash
  AND b.blockchain = s.blockchain
GROUP BY 1
ORDER BY 1