#pragma once
// Line-delimited JSON subprocess protocol.
// Request format (stdin):
//   {"type":"setup","truck":{"depth":D,"width":W,"height":H}}
//   {"type":"decide","current_box":{...},"placed_boxes":[...],"boxes_remaining":N,"density":D}
//   {"type":"teardown","final_density":D,"termination_reason":"..."}
//
// Response format (stdout, one line per request):
//   {"position":[x,y,z],"orientation_wxyz":[w,x,y,z],"stop":false}
//
// Errors: write to stderr. Do NOT write to stdout on error.
