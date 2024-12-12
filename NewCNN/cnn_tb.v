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

    // Clock and Reset
    reg clk;
    reg reset;
    
    // Control Signals
    reg conv_enable;
    reg pool_enable;
    reg fc_enable;
    reg input_valid;

    // Data Signals
    reg [15:0] input_data;
    wire [15:0] conv_output;
    wire [15:0] pool_output;
    wire [15:0] fc_output;
    
    // Status Signals
    wire conv_done;
    wire pool_done;
    wire fc_done;
    wire conv_output_valid;
    wire pool_output_valid;
    wire fc_output_valid;

    // Memory interface
    wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS)-1:0] conv_input_addr;
    wire [$clog2((INPUT_WIDTH-WINDOW_SIZE+1)*(INPUT_HEIGHT-WINDOW_SIZE+1)*NUM_NEURONS)-1:0] conv_output_addr;

    // Training Data
    reg [15:0] training_data [0:INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS-1];
    
    // Training Metrics
    reg [31:0] batch_loss;
    reg [31:0] epoch_loss;
    reg [7:0] accuracy;
    reg [15:0] learning_rate;
    reg [15:0] output_error;

    // Counters
    reg [3:0] current_epoch;
    reg [7:0] current_batch;
    
    // Module instantiations
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
        .conv_done(conv_done)
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
        .pool_done(pool_done)
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
        .fc_done(fc_done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Initialize training data
    integer i;
    initial begin
        for (i = 0; i < INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNELS; i = i + 1) begin
            training_data[i] = i[15:0];
        end
    end

    // Data feeding process
    always @(posedge clk) begin
        if (!reset && input_valid) begin
            input_data <= training_data[conv_input_addr];
        end
    end

    // Main test process
    initial begin
        // Initialize
        reset = 1;
        conv_enable = 0;
        pool_enable = 0;
        fc_enable = 0;
        input_valid = 0;
        learning_rate = 16'h0080;
        batch_loss = 0;
        epoch_loss = 0;
        accuracy = 0;
        current_epoch = 0;
        current_batch = 0;

        // Reset sequence
        #100;
        reset = 0;
        #100;

        // Training loop
        repeat (NUM_EPOCHS) begin
            $display("Starting Epoch %0d", current_epoch);
            
            repeat (BATCH_SIZE) begin
                // Convolution
                @(posedge clk);
                conv_enable = 1;
                input_valid = 1;
                @(posedge conv_done);
                @(posedge clk);
                conv_enable = 0;

                // Pooling
                @(posedge clk);
                pool_enable = 1;
                @(posedge pool_done);
                @(posedge clk);
                pool_enable = 0;

                // Fully Connected
                @(posedge clk);
                fc_enable = 1;
                @(posedge fc_done);
                @(posedge clk);
                fc_enable = 0;

                // Update metrics
                if (fc_output_valid) begin
                    batch_loss = batch_loss + fc_output;
                    accuracy = accuracy + 1;
                end

                current_batch = current_batch + 1;
            end

            current_epoch = current_epoch + 1;
            $display("Completed Epoch %0d", current_epoch);
        end

        #1000;
        $finish;
    end

    // Monitor progress
    always @(posedge clk) begin
        if (!reset) begin
            if (conv_output_valid) $display("Conv Output: %h", conv_output);
            if (pool_output_valid) $display("Pool Output: %h", pool_output);
            if (fc_output_valid) $display("FC Output: %h", fc_output);
        end
    end

endmodule