import nltk
import time
from collections import Counter, defaultdict

# Download texts #
nltk.download('brown')
nltk.download('gutenberg')
from nltk.corpus import brown, gutenberg


def make_4grams_from_corpus():
    print("Loading words from corpora")
    words = [w.lower() for w in brown.words()] + [w.lower() for w in gutenberg.words()]
    print(f"Loaded {len(words)} words")

    n = 4
    grams = []
    total_grams = len(words) - n + 1

    print(f"Generating {total_grams} N-grams...")

    start_time = time.time()

    for i in range(total_grams):
        fg_words = words[i:i + n]
        fg_text = " ".join(fg_words)
        fg_letters = "".join(fg_words)
        grams.append((fg_text, fg_letters, set(fg_letters), Counter(fg_letters)))

        if (i + 1) % 100000 == 0 or (i + 1) == total_grams:
            elapsed = time.time() - start_time
            progress = (i + 1) / total_grams * 100
            print(f"[{i + 1:,}/{total_grams:,}] ({progress:.1f}%) - {elapsed:.1f}s")

    print(f"Created {len(grams)} N-grams")
    return grams


def check_match_with_position(window, fg_letters_str, fg_letters_set, fg_counter):
    known = window.replace("*", "")
    if not known:
        return 0

    win_set = set(known)
    win_counter = Counter(known)

    if not win_set.issubset(fg_letters_set):
        return -1
    for letter, count in win_counter.items():
        if count > fg_counter.get(letter, 0):
            return -1

    base_score = sum(win_counter.values())
    position_bonus = 0

    for idx, ch in enumerate(window):
        if ch == "*":
            continue
        if idx < len(fg_letters_str) and fg_letters_str[idx] == ch:
            position_bonus += 1

    return base_score + 2 * position_bonus


def find_matches_with_positions(text, fourgrams):
    total = len(fourgrams)
    matches = []
    start_time = time.time()

    for idx, (fg_text, fg_letters_str, fg_letters_set, fg_counter) in enumerate(fourgrams, start=1):
        win_len = len(fg_letters_str)
        if win_len > len(text):
            continue

        for pos in range(len(text) - win_len + 1):
            window = text[pos:pos + win_len]
            score = check_match_with_position(window, fg_letters_str, fg_letters_set, fg_counter)
            if score > 0:
                matches.append((pos, fg_text, score))

        if idx % 50000 == 0 or idx == total:
            elapsed = time.time() - start_time
            progress = idx / total
            eta = elapsed / progress - elapsed
            print(f"[PROGRESS] {idx}/{total} ({progress * 100:.2f}%) | "
                  f"Time {elapsed:.1f}s | ETA ~{eta / 60:.1f} min")

    return matches


def reconstruct_text(matches, damaged_text_length):
    print("\n" + "=" * 50)
    print("TEXT RECONSTRUCTION")
    print("=" * 50)

    grouped_matches = defaultdict(list)
    for pos, fg, score in matches:
        grouped_matches[pos].append((fg, score))

    for pos in grouped_matches:
        grouped_matches[pos].sort(key=lambda x: -x[1])

    print("\n🔍 Analysis of key positions:")
    key_positions = sorted([pos for pos in grouped_matches.keys()
                            if grouped_matches[pos][0][1] >= 20])[:10]

    for pos in key_positions:
        best_match = grouped_matches[pos][0]
        print(f"  Position {pos:2d}: score {best_match[1]:2d} | '{best_match[0]}'")

    print(f"\n🔧 Greedy reconstruction:")

    selected_parts = []
    used_positions = set()

    all_positions = [(pos, grouped_matches[pos][0][1], grouped_matches[pos][0][0])
                     for pos in grouped_matches.keys()]
    all_positions.sort(key=lambda x: -x[1])

    for pos, score, text in all_positions:
        if score < 15:
            continue

        text_length = len(text.replace(" ", ""))
        if any(used_pos in range(pos, pos + text_length) for used_pos in used_positions):
            continue

        selected_parts.append((pos, text, score))
        used_positions.update(range(pos, pos + text_length))

    selected_parts.sort(key=lambda x: x[0])

    print("Selected parts:")
    for pos, text, score in selected_parts:
        print(f"  {pos:2d}: '{text}' (score: {score})")

    print(f"\n🎯 Final text assembly:")

    words = []

    for pos, text, score in selected_parts:
        text_words = text.split()
        for word in text_words:
            if word not in words or len(words) == 0:
                words.append(word)

    if words:
        reconstructed_sentence = " ".join(words)
        print(f"📝 Reconstructed sentence:")
        print(f"   '{reconstructed_sentence}'")

        reconstructed_no_spaces = reconstructed_sentence.replace(" ", "").lower()
        print(f"\n📊 Statistics:")
        print(f"   Length with spaces: {len(reconstructed_sentence)}")
        print(f"   Length without spaces: {len(reconstructed_no_spaces)}")
        print(f"   Original length: {damaged_text_length}")
        print(f"   Length match: {'✅' if len(reconstructed_no_spaces) == damaged_text_length else '❌'}")

        return reconstructed_sentence
    else:
        print("❌ Failed to reconstruct text")
        return None


def decode_text(damaged_text):
    print("🚀 STARTING TEXT DECODING")
    print("=" * 50)
    print(f"Damaged text: '{damaged_text}'")
    print(f"Length: {len(damaged_text)} characters")

    print("\n📚 Generating 4-grams from corpus...")
    fourgrams = make_4grams_from_corpus()
    print(f"Total 4-grams: {len(fourgrams)}")

    print("\n🔍 Finding matches...")
    matches = find_matches_with_positions(damaged_text, fourgrams)
    print(f"Found matches: {len(matches)}")

    result = reconstruct_text(matches, len(damaged_text))

    if result:
        print(f"\n✅ FINAL RESULT:")
        print(f"🎉 DECODED TEXT: '{result}'")
    else:
        print(f"\n❌ DECODING FAILED")

    return result


if __name__ == "__main__":
    damaged_text = (
        "Al*cew*sbegninnigtoegtver*triedofsitt*ngbyh*rsitsreonhtebnakandofh*vingnothi*gtodoonc*ortw*cesh*hdapee*edintoth*boo*h*rsiste*wasr*adnigbuti*hadnopictu*esorc*nve*sati*nsinitandwhatisth*useofab**kth*ughtAlic*withou*pic*u*esorco*versa*ions"
    )

    decoded = decode_text(damaged_text)