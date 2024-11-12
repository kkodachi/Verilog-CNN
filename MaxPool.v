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

    reg [9:0] i, j;
    reg [15:0] max; // max value within the window
    reg [9:0] row, col;
    reg [1:0] state;
    localparam IDLE = 2'b00, INIT = 2'b01, WORK = 2'b10;

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            row <= 0;
            col <= 0;
            i <= 0;
            j <= 0;
            max <= 0;
            state <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= INIT;
                    end
                end

                INIT: begin
                    i <= 0;
                    j <= 0;
                    row <= 0;
                    col <= 0;
                    max <= 0;
                    state <= WORK;
                end

                WORK: begin
                    if (featureMap[row*STRIDE + i][col*STRIDE + j] > max) begin
                        max <= featureMap[row*STRIDE + i][col*STRIDE + j];
                    end

                    if (j < KERNEL - 1) begin
                        j <= j + 1;
                    end else if (i < KERNEL - 1) begin
                        i <= i + 1;
                        j <= 0;
                    end else begin
                        i <= 0;
                        j <= 0;
                        pooled_output[row][col] <= max;
                        max <= 0;

                        if (col < (FM_width / STRIDE) - 1) begin
                            col <= col + 1;
                        end else begin
                            col <= 0;
                            if (row < (FM_height / STRIDE) - 1) begin
                                row <= row + 1;
                            end else begin
                                done <= 1;
                                state <= IDLE;
                            end
                        end
                    end
                end
            endcase
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

    reg [15:0] max;
    reg [9:0] max_x, max_y, i, j; // ind max value in featureMap window
    reg [9:0] row, col;
    reg [1:0] state;
    localparam IDLE = 2'b00, INIT = 2'b01, WORK = 2'b10;

    always @(posedge clk) begin
        if (rst) begin
            done <= 0;
            row <= 0;
            col <= 0;
            state <= IDLE;
            row <= 0;
            col <= 0;
            max_x <= 0;
            max_y <= 0;
            i <= 0;
            j <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) state <= INIT;
                end

                INIT: begin
                    // initialize input_error to 0 for each position in featureMap
                    input_error[i][j] <= 0;

                    if (j < FM_height - 1) begin
                        j <= j + 1;
                    end else if (i < FM_width - 1) begin
                        j <= 0;
                        i < i + 1;
                    end else begin
                        i <= 0;
                        j <= 0;
                        state <= WORK;
                    end
                end

                WORK: begin
                    if (featureMap[row*STRIDE + i][col*STRIDE + j] > max) begin
                        max = featureMap[row*STRIDE + i][col*STRIDE + j];
                        max_x = row * STRIDE + i;
                        max_y = col * STRIDE + j;
                    end

                    if (j < KERNEL - 1) begin
                        j <= j + 1;
                    end else if (i < KERNEL - 1) begin
                        i <= i + 1;
                        j <= 0;
                    end else begin
                        i <= 0;
                        j <= 0;
                        max_x <= 0;
                        max_y <= 0;
                        max <= 0;
                        input_error[max_x][max_y] <= input_error[max_x][max_y] + output_error[row][col];

                        if (col < (FM_width / STRIDE) - 1) begin
                            col <= col + 1;
                        end else begin
                            col <= 0;
                            if (row < (FM_height / STRIDE) - 1) begin
                                row <= row + 1;
                            end else begin
                                done <= 1;
                                state <= IDLE;
                            end
                        end
                    end
                end
            endcase
        end
    end
endmodule
