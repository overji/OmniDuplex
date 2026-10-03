Mode: real-time description. Provide at least two meaningful updates at natural
changes or evidence boundaries. Include the current state when useful, then new
information, not paraphrases of the same state. No forced initial silent interval.
Each response adds window_start and window_end (MM:SS), with
window_start <= window_end == trigger_time. Windows are chronological and do not
overlap (shared endpoints are allowed), and start no earlier than question_time.
Historical facts may contextualize the current state, but no future evidence is
allowed. Omit trigger objects: this mode continuously describes the chosen focus.
For action counting, fix counting_start across responses; it may precede the
question if the question explicitly asks for the running total from that origin.
