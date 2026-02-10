"""Custom pywikibot family for utaite.wiki (Miraheze)."""

from pywikibot import family


class Family(family.Family):
    """Family class for Utaite Wiki on Miraheze."""

    name = "utaitewiki"
    langs = {
        "en": "utaite.wiki",
    }

    def scriptpath(self, code):
        return "/w"

    def protocol(self, code):
        return "https"
