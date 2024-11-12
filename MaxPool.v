module MaxPool_Forward #(
    parameter FM_height = 62, // feature map height after Conv2d (64 - 3 + 1)
    parameter FM_width = 62, // feature map width after Conv2d (64 - 3 + 1)
    parameter KERNEL = 2, // pooling window size
    parameter STRIDE = 2 // size of stride for pooling operation
)(
    input clk,
    input rst,
    input start,
    input [15:0] featureMap [0:FM_width-1][0:FM_height-1], // featureMap from Conv2d
    output reg [15:0] pooled_output[0:FM_width/STRIDE-1][0:FM_height/STRIDE-1], // max values from each window
    output reg done
);

    integer i, j;
    reg [15:0] max; // max value within the window
    reg [9:0] row, col;

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            row <= 0;
            col <= 0;
        end else if (!done && start) begin
            max = 0;
            // find the max value in the window
            for (i = 0; i < KERNEL; i = i + 1) begin
                for (j = 0; j < KERNEL; j = j + 1) begin
                    if (featureMap[row*STRIDE + i][col*STRIDE + j] > max) begin
                        max = featureMap[row*STRIDE + i][col*STRIDE + j];
                    end
                end
            end
            
            pooled_output[row][col] <= max;

            // move window
            if (col < (FM_width / STRIDE) - 1) begin
                col <= col + 1;
            end else begin
                col <= 0;
                if (row < (FM_height / STRIDE) - 1) begin
                    row <= row + 1;
                end else begin
                    done <= 1;
                end
            end
        end
    end
endmodule

module MaxPool_backward #(
    parameter FM_height = 62, // feature map height after Conv2d (64 - 3 + 1)
    parameter FM_width = 62, // feature map width after Conv2d (64 - 3 + 1)
    parameter KERNEL = 2, // pooling window size
    parameter STRIDE = 2 // size of stride for pooling operation
)(
    input clk,
    input rst,
    input start,
    input [15:0] featureMap [0:FM_width-1][0:FM_height-1], // featureMap from Conv2d
    input [15:0] output_error [0:FM_width/STRIDE-1][0:FM_height/STRIDE-1], // errors from the MaxPool output
    output reg [15:0] input_error [0:FM_width-1][0:FM_height-1], // gradients for feature map
    output reg done
);

    integer i, j;
    reg [15:0] max;
    reg [9:0] max_x, max_y; // ind max value in featureMap window
    reg [9:0] row, col; 

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            row <= 0;
            col <= 0;
            // initialize input_error to 0 for each position in featureMap
            for (i = 0; i < FM_width; i = i + 1) begin
                for (j = 0; j < FM_height; j = j + 1) begin
                    input_error[i][j] <= 0;
                end
            end
        end

        else if (!done && start) begin
            // initialize max value and ind
            max = 0;
            max_x = 0;
            max_y = 0;

            // find max value and ind within window
            for (i = 0; i < KERNEL; i = i + 1) begin
                for (j = 0; j < KERNEL; j = j + 1) begin
                    if (featureMap[row*STRIDE + i][col*STRIDE + j] > max) begin
                        max = featureMap[row*STRIDE + i][col*STRIDE + j];
                        max_x = row * STRIDE + i;
                        max_y = col * STRIDE + j;
                    end
                end
            end

            // add output error to max location
            input_error[max_x][max_y] <= input_error[max_x][max_y] + output_error[row][col];

            // move window
            if (col < (FM_width / STRIDE) - 1) begin
                col <= col + 1;
            end else begin
                col <= 0;
                if (row < (FM_height / STRIDE) - 1) begin
                    row <= row + 1;
                end else begin
                    done <= 1;
                end
            end
        end
    end
endmodule
