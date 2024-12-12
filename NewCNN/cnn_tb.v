`timescale 1ns/1ps

module cnn_tb();
    // Previous parameters remain same...

    // Additional signals for backpropagation
    reg [15:0] output_error;
    wire [15:0] conv_error, pool_error;
    reg [FC_OUTPUT_SIZE-1:0] true_labels [0:BATCH_SIZE-1];
    reg [15:0] batch_loss;
    reg [7:0] accuracy;
    
    // Training counters
    reg [3:0] current_epoch;
    reg [7:0] current_batch;
    reg [15:0] sample_counter;

    // Original module instantiations with backprop connections
    conv2d #(
        // Previous parameters...
    ) conv_layer (
        // Previous connections...
        .output_error(conv_error),
        .learning_rate(learning_rate),
        .backprop_done(conv_backprop_done)
    );

    max_pool #(
        // Previous parameters...
    ) pool_layer (
        // Previous connections...
        .output_error(pool_error),
        .backprop_done(pool_backprop_done)
    );

    fully_connected #(
        // Previous parameters...
    ) fc_layer (
        // Previous connections...
        .output_error(output_error),
        .learning_rate(learning_rate),
        .backprop_done(fc_backprop_done)
    );

    // Test stimulus with training loop
    initial begin
        // Previous initialization...

        // Training loop
        for (current_epoch = 0; current_epoch < 5; current_epoch = current_epoch + 1) begin
            $display("\nStarting Epoch %0d", current_epoch + 1);
            
            for (current_batch = 0; current_batch < BATCH_SIZE; current_batch = current_batch + 1) begin
                // Forward pass
                conv_enable = 1;
                input_valid = 1;
                @(posedge conv_done);
                conv_enable = 0;

                pool_enable = 1;
                @(posedge pool_done);
                pool_enable = 0;

                fc_enable = 1;
                @(posedge fc_done);
                fc_enable = 0;

                // Calculate error and start backpropagation
                output_error = fc_output - true_labels[current_batch][fc_output_addr];
                batch_loss = batch_loss + (output_error * output_error) >> FIXED_POINT_BITS;

                // Wait for backpropagation completion
                @(posedge fc_backprop_done);
                @(posedge pool_backprop_done);
                @(posedge conv_backprop_done);

                // Update accuracy if prediction matches label
                if (fc_output == true_labels[current_batch][fc_output_addr])
                    accuracy = accuracy + 1;

                // Display batch progress
                $display("Batch %0d: Loss = %0d, Accuracy = %0d%%", 
                    current_batch,
                    batch_loss,
                    (accuracy * 100) / (current_batch + 1));
            end

            // Display epoch results
            $display("Epoch %0d Results:", current_epoch + 1);
            $display("Final Loss: %0d", batch_loss);
            $display("Final Accuracy: %0d%%", (accuracy * 100) / BATCH_SIZE);

            // Reset batch metrics
            batch_loss = 0;
            accuracy = 0;
        end

        $display("\nTraining Complete!");
        #1000;
        $finish;
    end

    // Error calculation and backpropagation monitoring
    always @(posedge clk) begin
        if (!reset && fc_output_valid) begin
            // Log prediction vs true label
            $display("Prediction: %d, True Label: %d", 
                    fc_output, 
                    true_labels[current_batch][fc_output_addr]);
            
            // Log error propagation
            if (|output_error) begin
                $display("Error at layer: FC=%d, Pool=%d, Conv=%d",
                        output_error, pool_error, conv_error);
            end
        end
    end
endmodule