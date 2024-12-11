# Setup waves for CNN training visualization
vsim -t 1ps work.cnn_tb

# Add basic control signals
add wave -divider "Control"
add wave -color Yellow /cnn_tb/clk
add wave -color Yellow /cnn_tb/reset
add wave -color Yellow /cnn_tb/conv_enable
add wave -color Yellow /cnn_tb/pool_enable
add wave -color Yellow /cnn_tb/fc_enable
add wave -color Yellow /cnn_tb/current_epoch
add wave -color Yellow /cnn_tb/current_batch

# Add training metrics with analog display
add wave -divider "Training Metrics"
add wave -format Analog-Step -height 100 -min 0 -max 100 -color Blue /cnn_tb/epoch_accuracy_history
add wave -format Analog-Step -height 100 -min 0 -max 1000 -color Red /cnn_tb/batch_loss
add wave -format Analog-Step -height 100 -min 0 -max 100 -color Green /cnn_tb/correct_predictions
add wave /cnn_tb/total_samples

# Add layer outputs
add wave -divider "Layer Outputs"
add wave /cnn_tb/conv_layer_output
add wave /cnn_tb/pool_layer_output
add wave /cnn_tb/fc_layer_output

# Configure wave window
configure wave -namecolwidth 220
configure wave -valuecolwidth 100
configure wave -signalnamewidth 1
configure wave -timelineunits ns

# Run simulation
run -all

# Zoom wave window to show all
wave zoom full