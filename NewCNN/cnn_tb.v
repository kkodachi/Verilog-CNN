`timescale 1ns/1ps

module cnn_tb();
    // Architecture Parameters
    parameter INPUT_WIDTH = 64;
    parameter INPUT_HEIGHT = 64;
    parameter INPUT_CHANNELS = 1;
    parameter WINDOW_SIZE = 3;
    parameter NUM_NEURONS = 30;
    parameter POOL_STRIDE = 2;
    parameter FC_OUTPUT_SIZE = 10;
    parameter FIXED_POINT_BITS = 8;

    // Training Parameters
    parameter NUM_EPOCHS = 5;
    parameter BATCH_SIZE = 32;
    parameter NUM_BATCHES = 10;

    // Clock and Reset
    reg clk;
    reg reset;
    
    // Control Signals
    reg conv_enable, pool_enable, fc_enable;
    wire conv_done, pool_done, fc_done;
    wire conv_backprop_done, pool_backprop_done, fc_backprop_done;

    // Data Signals
    reg [15:0] input_data;
    wire [15:0] conv_output;
    wire [15:0] pool_output;
    wire [15:0] fc_output;
    reg [15:0] learning_rate;

    // Training Data Storage
    reg [15:0] training_data [0:INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS-1];
    reg [FC_OUTPUT_SIZE-1:0] true_labels [0:BATCH_SIZE-1];
    reg [15:0] network_outputs [0:FC_OUTPUT_SIZE-1];

    // Memory Interface Signals
    wire conv_output_valid, pool_output_valid, fc_output_valid;
    reg input_valid;
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] conv_input_addr;
    wire [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] conv_output_addr;
    wire [$clog2((INPUT_WIDTH/POOL_STRIDE)*(INPUT_HEIGHT/POOL_STRIDE)*NUM_NEURONS)-1:0] pool_output_addr;
    wire [$clog2(FC_OUTPUT_SIZE)-1:0] fc_output_addr;

    // Error and Training Metrics
    reg [15:0] output_error;
    wire [15:0] conv_error, pool_error;
    reg [31:0] batch_loss;
    reg [31:0] epoch_loss;
    reg [7:0] accuracy;
    reg [3:0] current_epoch;
    reg [7:0] current_batch;
    reg [15:0] sample_counter;

    // Instantiate CNN modules
    conv2d #(
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .WINDOW_SIZE(WINDOW_SIZE),
        .NUM_NEURONS(NUM_NEURONS)
    ) conv_layer (
        .clk(clk),
        .reset(reset),
        .enable(conv_enable),
        .input_data(input_data),
        .input_addr(conv_input_addr),
        .input_valid(input_valid),
        .feature_map(conv_output),
        .output_addr(conv_output_addr),
        .output_valid(conv_output_valid),
        .conv_done(conv_done),
        .output_error(conv_error),
        .learning_rate(learning_rate),
        .backprop_done(conv_backprop_done)
    );

    max_pool #(
        .INPUT_WIDTH(INPUT_WIDTH-WINDOW_SIZE+1),
        .INPUT_HEIGHT(INPUT_HEIGHT-WINDOW_SIZE+1),
        .INPUT_CHANNELS(NUM_NEURONS),
        .STRIDE(POOL_STRIDE)
    ) pool_layer (
        .clk(clk),
        .reset(reset),
        .enable(pool_enable),
        .input_data(conv_output),
        .input_valid(conv_output_valid),
        .pooled_output(pool_output),
        .output_addr(pool_output_addr),
        .output_valid(pool_output_valid),
        .pool_done(pool_done),
        .output_error(pool_error),
        .backprop_done(pool_backprop_done)
    );

    fully_connected #(
        .INPUT_SIZE((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS/(POOL_STRIDE*POOL_STRIDE)),
        .OUTPUT_SIZE(FC_OUTPUT_SIZE)
    ) fc_layer (
        .clk(clk),
        .reset(reset),
        .enable(fc_enable),
        .input_data(pool_output),
        .input_valid(pool_output_valid),
        .output_data(fc_output),
        .output_addr(fc_output_addr),
        .output_valid(fc_output_valid),
        .fc_done(fc_done),
        .output_error(output_error),
        .learning_rate(learning_rate),
        .backprop_done(fc_backprop_done)
    );

    // Clock Generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Monitor input data
    always @(posedge clk) begin
        if (!reset && input_valid) begin
            input_data <= training_data[conv_input_addr];
        end
    end

    // Training Process
    initial begin
        // Initialize signals
        reset = 1;
        conv_enable = 0;
        pool_enable = 0;
        fc_enable = 0;
        input_valid = 0;
        learning_rate = 16'h0080; // 0.5 in fixed point
        current_epoch = 0;
        current_batch = 0;
        sample_counter = 0;
        batch_loss = 0;
        epoch_loss = 0;
        accuracy = 0;

        // Generate training data
        for (integer i = 0; i < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS; i = i + 1) begin
            training_data[i] = i[15:0]; // Sequential test data
        end

        // Generate random labels
        for (integer i = 0; i < BATCH_SIZE; i = i + 1) begin
            true_labels[i] = 1 << ($random % FC_OUTPUT_SIZE); // One-hot encoding
        end

        // Release reset
        #100;
        reset = 0;
        #100;

        // Training loop
        current_epoch = 0;
        while (current_epoch < NUM_EPOCHS) begin
            $display("\nStarting Epoch %0d", current_epoch + 1);
            epoch_loss = 0;
            
            current_batch = 0;
            while (current_batch < BATCH_SIZE) begin
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

                // Calculate error and update weights
                output_error = fc_output - true_labels[current_batch][fc_output_addr];
                batch_loss = batch_loss + (output_error * output_error) >>> FIXED_POINT_BITS;

                // Wait for backpropagation
                @(posedge conv_backprop_done);
                @(posedge pool_backprop_done);
                @(posedge fc_backprop_done);

                // Update accuracy
                if (fc_output == true_labels[current_batch][fc_output_addr])
                    accuracy = accuracy + 1;

                // Display progress
                $display("Batch %0d: Loss = %0d, Accuracy = %0d%%", 
                    current_batch,
                    batch_loss,
                    (accuracy * 100) / (current_batch + 1));
                
                current_batch = current_batch + 1;
            end

            // Display epoch results
            $display("Epoch %0d Complete", current_epoch + 1);
            $display("Average Loss: %0d", batch_loss/BATCH_SIZE);
            $display("Final Accuracy: %0d%%", (accuracy * 100) / BATCH_SIZE);

            // Reset metrics for next epoch
            batch_loss = 0;
            accuracy = 0;
            current_epoch = current_epoch + 1;
        end

        // Training complete
        $display("\nTraining Complete!");
        #1000;
        $finish;
    end

    // Generate waveform file
    initial begin
        $dumpfile("cnn_training.vcd");
        $dumpvars(0, cnn_tb);
    end

endmodule