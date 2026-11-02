""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# MFA IPA phone inventory for English and Indian English alignment compatibility:
_mfa_phones_list = [
    'a', 'aj', 'aw', 'aː', 'b', 'bʲ', 'c', 'cʰ', 'cʷ', 'd', 'dʒ', 'dʲ', 'd̪', 'e', 'ej',
    'eː', 'f', 'fʲ', 'fʷ', 'h', 'i', 'iː', 'j', 'k', 'kp', 'kʰ', 'kʷ', 'l', 'm', 'mʲ',
    'm̩', 'n', 'n̩', 'o', 'ow', 'oː', 'p', 'pʰ', 'pʲ', 'pʷ', 's', 't', 'tʃ', 'tʰ', 'tʲ',
    'tʷ', 't̪', 'u', 'uː', 'v', 'vʲ', 'vʷ', 'w', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ',
    'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔj', 'ɖ', 'ə', 'əw', 'ɚ', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɝ', 'ɟ',
    'ɟʷ', 'ɡ', 'ɡb', 'ɡʷ', 'ɪ', 'ɫ', 'ɫ̩', 'ɱ', 'ɲ', 'ɹ', 'ɾ', 'ɾʲ', 'ɾ̃', 'ʃ', 'ʈ',
    'ʈʲ', 'ʈʷ', 'ʉ', 'ʉː', 'ʊ', 'ʋ', 'ʎ', 'ʒ', 'ʔ', 'θ'
]
_mfa = ["@" + s for s in _mfa_phones_list]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _mfa
    + _silences
)
