DEV=mlx5_0; PORT=1; INT=0.1
rx_prev=$(< /sys/class/infiniband/$DEV/ports/$PORT/counters/port_rcv_data)
tx_prev=$(< /sys/class/infiniband/$DEV/ports/$PORT/counters/port_xmit_data)
t_prev=$(date +%s%N)
while sleep $INT; do
  rx=$(< /sys/class/infiniband/$DEV/ports/$PORT/counters/port_rcv_data)
  tx=$(< /sys/class/infiniband/$DEV/ports/$PORT/counters/port_xmit_data)
  t_now=$(date +%s%N); dt_ns=$((t_now - t_prev))
  rx_gbps=$(awk -v a="$rx_prev" -v b="$rx" -v ns="$dt_ns" 'BEGIN{print ((b-a)*4*8)/(ns/1e9)/1e9}')
  tx_gbps=$(awk -v a="$tx_prev" -v b="$tx" -v ns="$dt_ns" 'BEGIN{print ((b-a)*4*8)/(ns/1e9)/1e9}')
  printf "RX: %.2f Gbit/s   TX: %.2f Gbit/s\n" "$rx_gbps" "$tx_gbps"
  rx_prev=$rx; tx_prev=$tx; t_prev=$t_now
done