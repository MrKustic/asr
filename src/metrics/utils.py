# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    assert target_text
    target_len = len(target_text)

    prev_levenstein_dist = [j for j in range(target_len + 1)]
    for i in range(len(predicted_text)):
        current_levenstein_dist = [0] * (target_len + 1)
        current_levenstein_dist[0] = i + 1
        for j in range(target_len):
            if predicted_text[i] == target_text[j]:
                current_levenstein_dist[j + 1] = prev_levenstein_dist[j]
            else:
                current_levenstein_dist[j + 1] = (
                    min(
                        current_levenstein_dist[j],
                        prev_levenstein_dist[j + 1],
                        prev_levenstein_dist[j],
                    )
                    + 1
                )
        prev_levenstein_dist = current_levenstein_dist

    return prev_levenstein_dist[-1] / target_len


def calc_wer(target_text, predicted_text) -> float:
    return calc_cer(target_text.split(), predicted_text.split())
