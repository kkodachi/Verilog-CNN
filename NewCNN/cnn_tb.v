`timescale 1ns/1ps

module cnn_tb();
    // Parameters
    parameter INPUT_WIDTH = 64;
    parameter INPUT_HEIGHT = 64;
    parameter INPUT_CHANNELS = 1;
    parameter WINDOW_SIZE = 3;
    parameter NUM_NEURONS = 30;
    parameter POOL_STRIDE = 2;
    parameter FC_OUTPUT_SIZE = 10;
    parameter BATCH_SIZE = 32;
    parameter NUM_EPOCHS = 5;

    // Clock and reset
    reg clk;
    reg reset;
    
    // Control signals
    reg conv_enable, pool_enable, fc_enable;
    wire conv_done, pool_done, fc_done;
    wire conv_backprop_done, pool_backprop_done, fc_backprop_done;

    // Memory interfaces
    reg [15:0] input_data [0:INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS-1];
    reg [FC_OUTPUT_SIZE-1:0] true_labels [0:BATCH_SIZE-1];
    wire [15:0] conv_output, pool_output, fc_output;

    // Training metrics
    reg [31:0] batch_loss;
    reg [31:0] epoch_loss;
    reg [7:0] accuracy;
    reg [15:0] learning_rate;

    // Addresses and control
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] conv_input_addr;
    wire [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] conv_output_addr;
    wire conv_output_valid;
    reg conv_input_valid;

    // Current progress
    reg [3:0] current_epoch;
    reg [7:0] current_batch;
    reg [15:0] sample_counter;

    // Error signals
    wire [15:0] conv_error, pool_error, fc_error;
    reg [15:0] output_error;

    // Loop variables
    reg [31:0] i;
    
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
        .input_data(input_data[conv_input_addr]),
        .input_addr(conv_input_addr),
        .input_valid(conv_input_valid),
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
        .output_valid(fc_output_valid),
        .fc_done(fc_done),
        .output_error(fc_error),
        .learning_rate(learning_rate),
        .backprop_done(fc_backprop_done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Initialize and run training
    initial begin
        // Initialize signals
        reset = 1;
        conv_enable = 0;
        pool_enable = 0;
        fc_enable = 0;
        conv_input_valid = 0;
        learning_rate = 16'h0080; // 0.5 in fixed point
        
        // Initialize counters
        current_epoch = 0;
        current_batch = 0;
        sample_counter = 0;
        batch_loss = 0;
        epoch_loss = 0;
        accuracy = 0;

        // Generate random training data
        i = 0;
        while (i < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS) begin
            input_data[i] = $random;
            i = i + 1;
        end
        
        // Generate random labels
        i = 0;
        while (i < BATCH_SIZE) begin
            true_labels[i] = 1 << ($random % FC_OUTPUT_SIZE); // One-hot encoding
            i = i + 1;
        end

        // Release reset
        #100 reset = 0;
        #100 conv_input_valid = 1;

        // Training loop
        current_epoch = 0;
        while (current_epoch < NUM_EPOCHS) begin
            $display("\nStarting Epoch %0d", current_epoch + 1);
            epoch_loss = 0;
            
            current_batch = 0;
            while (current_batch < BATCH_SIZE) begin
                // Forward pass
                conv_enable = 1;
                @(posedge conv_done) conv_enable = 0;
                
                pool_enable = 1;
                @(posedge pool_done) pool_enable = 0;
                
                fc_enable = 1;
                @(posedge fc_done) fc_enable = 0;

                // Calculate error and update weights
                output_error = fc_output - true_labels[current_batch][fc_layer.output_addr];
                epoch_loss = epoch_loss + output_error;

                // Wait for backpropagation
                @(posedge conv_backprop_done);
                @(posedge pool_backprop_done);
                @(posedge fc_backprop_done);

                // Display progress
                $display("Batch %0d: Loss = %0d", current_batch, output_error);
                
                current_batch = current_batch + 1;
            end

            // Display epoch results
            $display("Epoch %0d completed. Average Loss: %0d", 
                    current_epoch + 1, epoch_loss/BATCH_SIZE);
            
            current_epoch = current_epoch + 1;
        end

        // Training complete
        $display("\nTraining Complete!");
        $display("Final Loss: %0d", epoch_loss/BATCH_SIZE);
        
        #1000 $finish;
    end

    // Calculate accuracy
    always @(posedge fc_done) begin
        if (!reset) begin
            if (fc_output == true_labels[current_batch][fc_layer.output_addr])
                accuracy <= accuracy + 1;
        end
    end

    // Generate VCD file
    initial begin
        $dumpfile("cnn_training.vcd");
        $dumpvars(0, cnn_tb);
    end

endmodule