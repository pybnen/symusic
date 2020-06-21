# most methods/functionality is taken from magenta
# https://github.com/magenta/magenta
# see: https://github.com/magenta/magenta/blob/master/magenta/music/sequences_lib.py
import numpy as np
import copy

from symusic.music import settings
from symusic.music.note_container import Tempo, NoteContainer, TimeSignature


class MultipleTempoError(Exception):
    pass


class BadTimeSignatureError(Exception):
    pass


class MultipleTimeSignatureError(Exception):
    pass


class NegativeTimeError(Exception):
    pass


def _extract_subsequences(nc, split_times):
    if len(split_times) < 2:
        raise ValueError('Must provide at least a start and end time.')
    if any(t1 > t2 for t1, t2 in zip(split_times[:-1], split_times[1:])):
        raise ValueError('Split times must be sorted.')
    if any(time >= nc.total_time for time in split_times[:-1]):
        raise ValueError('Cannot extract subsequence past end of sequence.')

    subsequences = [NoteContainer(total_time=0.0, instrument_infos=copy.copy(nc.instrument_infos))
                    for _ in range(len(split_times) - 1)]

    # Extract notes into subsequences.
    subsequence_index = -1
    for note in sorted(nc.notes, key=lambda note: note.start_time):
        if note.start_time < split_times[0]:
            # ignore notes before start of first subsequence
            continue

        while (subsequence_index < len(split_times) - 1 and
               note.start_time >= split_times[subsequence_index + 1]):
            subsequence_index += 1

        if subsequence_index == len(split_times) - 1:
            break

        # set start/end time relative to subsequence start
        note_cpy = copy.copy(note)
        note_cpy.start_time = note_cpy.start_time - split_times[subsequence_index]
        note_cpy.end_time = min(note.end_time, split_times[subsequence_index + 1]) - split_times[subsequence_index]
        subsequences[subsequence_index].notes.append(note_cpy)
        if note_cpy.end_time > subsequences[subsequence_index].total_time:
            subsequences[subsequence_index].total_time = note_cpy.end_time

    # Extract time signatures and tempos
    events_by_type = [nc.time_signatures, nc.tempos]
    new_event_containers = [[s.time_signatures for s in subsequences],
                            [s.tempos for s in subsequences]]

    for events, containers in zip(events_by_type, new_event_containers):
        previous_event = None
        subsequence_index = -1
        for event in sorted(events, key=lambda event: event.time):
            if event.time <= split_times[0]:
                previous_event = event
                continue
            while (subsequence_index < len(split_times) - 1 and
                   event.time > split_times[subsequence_index + 1]):
                subsequence_index += 1
                if subsequence_index == len(split_times) - 1:
                    break
                if previous_event is not None:
                    # Add state event to the beginning of the subsequence.
                    containers[subsequence_index].extend([copy.copy(previous_event)])
                    containers[subsequence_index][-1].time = 0.0
            if subsequence_index == len(split_times) - 1:
                break
            # Only add the event if it's actually inside the subsequence (and not on
            # the boundary with the next one).
            if event.time < split_times[subsequence_index + 1]:
                containers[subsequence_index].extend([copy.copy(event)])
                containers[subsequence_index][-1].time -= split_times[subsequence_index]
            previous_event = event
        # Add final state event to the beginning of all remaining subsequences.
        while subsequence_index < len(split_times) - 2:
            subsequence_index += 1
            if previous_event is not None:
                containers[subsequence_index].extend([copy.copy(previous_event)])
                containers[subsequence_index][-1].time = 0.0

    return subsequences


def split_on_time_signature_tempo_change(nc):
    time_signatures_and_tempos = sorted(nc.time_signatures + nc.tempos, key=lambda t: t.time)

    cur_denom = settings.DEFAULT_DENOMINATOR
    cur_num = settings.DEFAULT_NUMERATOR
    cur_qpm = settings.DEFAULT_QPM

    valid_split_times = [0.0]
    for event in time_signatures_and_tempos:
        if isinstance(event, Tempo):
            if np.isclose(cur_qpm, event.qpm):
                # tempo didn't change
                continue
        else:
            if cur_denom == event.denominator and cur_num == event.numerator:
                # time signature didn't change
                continue

        if event.time > valid_split_times[-1]:
            valid_split_times.append(event.time)

        if isinstance(event, Tempo):
            cur_qpm = event.qpm
        else:
            cur_num = event.numerator
            cur_denom = event.denominator

    # Handle the final subsequence.
    if nc.total_time > valid_split_times[-1]:
        valid_split_times.append(nc.total_time)

    if len(valid_split_times) > 1:
        return _extract_subsequences(nc, valid_split_times)
    else:
        return []


def _is_power_of_2(x):
    return x and not x & (x - 1)


def quantize_to_step(unquantized_seconds, steps_per_second, quantize_cutoff=settings.QUANTIZE_CUTOFF):
    unquantized_steps = unquantized_seconds * steps_per_second
    return int(unquantized_steps + (1 - quantize_cutoff))


def _quantize_notes(note_container, steps_per_second):
    for note in note_container.notes:
        # Quantize the start and end times of the note.
        note.quantized_start_step = quantize_to_step(note.start_time, steps_per_second)
        note.quantized_end_step = quantize_to_step(note.end_time, steps_per_second)

        if note.quantized_end_step == note.quantized_start_step:
            note.quantized_end_step += 1

        # Do not allow notes to start or end in negative time.
        if note.quantized_start_step < 0 or note.quantized_end_step < 0:
            raise NegativeTimeError('Got negative note time: start_step = %s, end_step = %s' %
                                    (note.quantized_start_step, note.quantized_end_step))

        # Extend quantized sequence if necessary.
        if note.quantized_end_step > note_container.total_quantized_steps:
            note_container.total_quantized_steps = note.quantized_end_step


def quantize_note_container(nc, steps_per_quarter):
    qnc = copy.deepcopy(nc)

    qnc.steps_per_quarter = steps_per_quarter

    if qnc.time_signatures:
        time_signatures = sorted(qnc.time_signatures, key=lambda ts: ts.time)
        # There is an implicit 4/4 time signature at 0 time. So if the first time
        # signature is something other than 4/4 and it's at a time other than 0,
        # that's an implicit time signature change.
        if time_signatures[0].time != 0 and not (
                time_signatures[0].numerator == 4 and
                time_signatures[0].denominator == 4):
            raise MultipleTimeSignatureError(
                'NoteSequence has an implicit change from initial 4/4 time '
                'signature to %d/%d at %.2f seconds.' %
                (time_signatures[0].numerator, time_signatures[0].denominator,
                 time_signatures[0].time))

        for time_signature in time_signatures[1:]:
            if (time_signature.numerator != qnc.time_signatures[0].numerator or
                    time_signature.denominator != qnc.time_signatures[0].denominator):
                raise MultipleTimeSignatureError(
                    'NoteSequence has at least one time signature change from %d/%d to '
                    '%d/%d at %.2f seconds.' %
                    (time_signatures[0].numerator, time_signatures[0].denominator,
                     time_signature.numerator, time_signature.denominator,
                     time_signature.time))

        # Make it clear that there is only 1 time signature and it starts at the beginning.
        qnc.time_signatures[0].time = 0
        del qnc.time_signatures[1:]
    else:
        time_signature = TimeSignature(time=0.0, numerator=settings.DEFAULT_NUMERATOR,
                                       denominator=settings.DEFAULT_DENOMINATOR)
        qnc.time_signatures.append(time_signature)

    if not _is_power_of_2(qnc.time_signatures[0].denominator):
        raise BadTimeSignatureError('Denominator is not a power of 2. Time signature: %d/%d' %
                                    (qnc.time_signatures[0].numerator, qnc.time_signatures[0].denominator))

    if qnc.time_signatures[0].numerator == 0:
        raise BadTimeSignatureError('Numerator is 0. Time signature: %d/%d' %
                                    (qnc.time_signatures[0].numerator, qnc.time_signatures[0].denominator))

    if qnc.tempos:
        tempos = sorted(qnc.tempos, key=lambda t: t.time)
        # There is an implicit 120.0 qpm tempo at 0 time. So if the first tempo is
        # something other that 120.0 and it's at a time other than 0, that's an
        # implicit tempo change.
        if tempos[0].time != 0 and tempos[0].qpm != settings.DEFAULT_QPM:
            raise MultipleTempoError(
                'NoteSequence has an implicit tempo change from initial %.1f qpm to '
                '%.1f qpm at %.2f seconds.' % (settings.DEFAULT_QPM, tempos[0].qpm, tempos[0].time))

        for tempo in tempos[1:]:
            if tempo.qpm != qnc.tempos[0].qpm:
                raise MultipleTempoError(
                    'NoteSequence has at least one tempo change from %.1f qpm to %.1f '
                    'qpm at %.2f seconds.' % (tempos[0].qpm, tempo.qpm, tempo.time))

        # Make it clear that there is only 1 tempo and it starts at the beginning.
        qnc.tempos[0].time = 0
        del qnc.tempos[1:]
    else:
        tempo = Tempo(time=0.0, qpm=settings.DEFAULT_QPM)
        qnc.tempos.append(tempo)

    # Compute quantization steps per second
    steps_per_second = qnc.tempos[0].qpm / 60.0 * steps_per_quarter
    qnc.total_quantized_steps = quantize_to_step(qnc.total_time, steps_per_second)
    _quantize_notes(qnc, steps_per_second)

    return qnc


def steps_per_bar_in_quantized_container(qnc):
    quarters_per_beat = 4.0 / qnc.time_signatures[0].denominator
    quarters_per_bar = quarters_per_beat * qnc.time_signatures[0].numerator

    return qnc.steps_per_quarter * quarters_per_bar
